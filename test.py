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
import torch.nn as nn


LABELS_DIR = r'data\vcdb\category.csv'

num_frames = config.d_frames
num_segs = config.num_segments
k_random = config.k_random
num_class = config.num_classes
img_feature_dim = config.img_feature_dim
image_size = config.image_size

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),     
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
model = MultiScaleTRN(img_feature_dim, num_frames, num_segs, k_random, num_class)
state_dict = torch.load(r'output\vcdb_model.pth')

model.load_state_dict(state_dict, strict=False)
# load the model and set it to evaluation mode
model.eval()
# model = nn.DataParallel(model)
model.to(DEVICE)

def load_image_data(data_path):
    image_path_list = os.listdir(data_path)
    frames_num = len(image_path_list)
    image_list = list()

    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    # Returns num evenly spaced samples, calculated over the interval [start, stop].
    # Ouput: samples, step -> put dim=0 is to get only samples
    sample = np.linspace(0, frames_num-1, num_segs, endpoint=True, retstep=True, dtype=int)[0]
    for i in sample:
        img_path = os.path.join(data_path, image_path_list[i])
        img = Image.open(img_path)
        img = transform(img)
        image_list.append(img)
    image = torch.stack(image_list)
    return image

 
'''
Predict the extracted frames
'''
if __name__ == '__main__':    

    # Load class names
    df = pd.read_csv(LABELS_DIR)
    classes = df['classIDx'].values.tolist()

    test_file = r'data\vcdb\test.csv'
    test_df = pd.read_csv(test_file)
    ROOT_PATH = r'data\vcdb\test'

    '''
    test model with extracted frames
    '''
    rows, cols = test_df.shape
    total_video = rows
    correct_nums = 0

    for item in test_df.itertuples():
        video_name = item.video_name.split('.')[0]        
        gt_label = item.classIDx
        DATA_DIR = os.path.join(ROOT_PATH, video_name)
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

        if(pred_label == gt_label):
            correct_nums += 1

        print("[INFO] groundtruth: {}, predicted label: {}, scores: {:.4f}".format(gt_label, pred_label, scores))
    
    print("[INFO] total videos: {}, correct: {:.4f}".format(total_video, correct_nums))
    


