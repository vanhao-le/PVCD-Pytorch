from imutils.video import FileVideoStream
from imutils.video import FPS
import imutils
import cv2
import pandas as pd
import torch
from torchvision import transforms, models, datasets
from torch.autograd import Variable
from model.TRN import TRN, MultiScaleTRN
import numpy as np
import os
from PIL import Image
import chip.config as config
from torch.nn import functional as F



LABELS_DIR = r'data\ucf5\category.txt'

num_frames = config.d_frames
num_segs = config.num_segments
num_class = config.num_classes
img_feature_dim = config.img_feature_dim
image_size = config.image_size

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),     
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
model = MultiScaleTRN(img_feature_dim, num_frames, num_segs, num_class)
state_dict = torch.load(r'output\best_model.pth')

model.load_state_dict(state_dict, strict=False)
# load the model and set it to evaluation mode
model.eval()

model.to(DEVICE)

def load_image_data(data_path):
    image_path_list = os.listdir(data_path)
    frames_num = len(image_path_list)
    image_list = list()
    sample = np.linspace(0, frames_num-1, num_segs, endpoint=True, retstep=True, dtype=int)[0]
    for i in sample:
        img_path = os.path.join(data_path, image_path_list[i])
        img = Image.open(img_path)
        img = transform(img)
        image_list.append(img)
    image = torch.stack(image_list)
    return image

def image_demo():
    # Load class names
    classes = list(np.loadtxt(LABELS_DIR, dtype=np.str_, delimiter=' '))
    
    '''
    test model with extracted frames
    '''
    # load folder
    ROOT_PATH = r'D:\PhD_Program\Deep-Learning\PVCD-Pytorch\data\ucf5\test'
    Video_name = 'v_ShavingBeard_g01_c02'
    DATA_DIR = os.path.join(ROOT_PATH, Video_name)
    
    # Tensor [num_segments, num_channel, width, heigth]
    dummy_input = load_image_data(DATA_DIR)

    # add batch_size dimension
    # Tensor [batch_size, num_segments, num_channel, width, heigth]
    dummy_input = torch.unsqueeze(dummy_input, dim=0).cuda()
    outputs = model(dummy_input)
    
    # find the class label index with the maximum probability
    class_prob = F.softmax(outputs, dim=1)
    
    # get most probable class and its probability:
    scores, idX = torch.max(class_prob, dim=1)
    # get class names    
    scores = scores.cpu().detach().numpy()[0]
    pred_label = classes[idX]

    print("[INFO] predicted label: {}, scores: {:.4f}".format(pred_label, scores))

def put_lable(frame, label):
    """
    Add iterations per second text to lower-left corner of a frame.
    opencv uses B G R order
    """
    cv2.putText(frame, "Predicted: " + label,
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255))
    return frame


'''
load video
'''
def video_demo():
    # Load class names
    classes = list(np.loadtxt(LABELS_DIR, dtype=np.str_, delimiter=' '))

    # created a *threaded *video stream, allow the camera senor to warmup,
    # and start the FPS counter
    print("[INFO] sampling THREADED frames from video...")
    
    # load video
    ROOT_PATH = r'D:\PhD_Program\Deep-Learning\Datasets\ucf101_top5\test'
    Video_name = 'v_TennisSwing_g06_c02.avi'
    DATA_DIR = os.path.join(ROOT_PATH, Video_name)

    vs = FileVideoStream(path=DATA_DIR).start()
    fps = FPS().start()

    pred_label = "";
    buffer_frames = []
    best_lable = ""
    best_score = 0
    # loop over some frames...this time using the threaded stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()

        view_frame = put_lable(frame, pred_label)
        

        # if the frame was not grabbed, then we have reached the end
        if frame is None:
            break

        # crop_img = cv2.resize(frame, (image_size, image_size))
        input_pill = Image.fromarray(frame)
        input_pill = transform(input_pill)
        
        cv2.imshow("Frame", view_frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        '''
        The size of queue == num_segments
        add frames to the queue if the queue is not full
        when the queue is full of frames then take first d_frames to feed the model
        '''
        if (len(buffer_frames) < num_segs):
            buffer_frames.append(input_pill)
        else:
            input_frames = torch.stack(buffer_frames)
            # input_frame shape
            # print("Before batch_size",input_frames.shape)
                      
            input = torch.unsqueeze(input_frames, dim=0).cuda()
            # print("Input shape:", input.shape)

            with torch.no_grad():
                outputs = model(input)

                # find the class label index with the maximum probability
                # h_x = torch.mean(F.softmax(outputs, dim=1), dim=0).data
                
                h_x = F.softmax(outputs, dim=1)

                # get most probable class and its probability:
                # probs, idx = h_x.sort(dim=0, descending=True)

                probs, idx = torch.max(h_x, dim=1)

                # scores = probs[0].cpu().detach().numpy()

                scores = probs.cpu().detach().numpy()[0]
                pred_label = classes[idx]
                # print("[INFO] predicted label: {}, scores: {:.4f}".format(pred_label, scores))

                if (scores > best_score):
                    best_score = scores
                    best_lable = pred_label
                
            
            '''
            list indices
            thislist = ["apple", "banana", "cherry"]
            
            The search will start at index 2 (included) and end at index 5 (not included)
            thislist[0:2] -> ['apple', 'banana']            
            thislist[:-1] -> ['apple', 'banana']
            thislist[1:]  -> ['banana', 'cherry']
            thislist[-1]  -> cherry
            '''
            # Remember that the last item has the index -1
            buffer_frames[:-1] = buffer_frames[1:]
            # Negative indexing means start from the end
            buffer_frames[-1] = input_pill
            # check to see if the frame should be displayed to our screen

        # update the FPS counter
        fps.update()

    print(print("[INFO] best label: {}, scores: {:.4f}".format(best_lable, best_score)))
    return

if __name__ == '__main__':    
    '''
    Predict the extracted frames
    '''
    # image_demo()

    '''
    Predict an input video
    '''

    video_demo()


