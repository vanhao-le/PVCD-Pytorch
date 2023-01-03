import torch
import numpy as np

# img_i = torch.rand(1,3,224,224)
# img_j = torch.rand(1,3,224,224)

# img = list()

# img.extend(img_i)
# img.extend(img_j)


# image_lst = torch.stack(img)

# N, T, C, H, W = image_lst.shape[:5]

# input_data = image_lst.reshape(-1, C, H, W)

# print(image_lst.shape)
# print(input_data.shape)
# num_frames = 8
# for i in range(num_frames):
#     print(i)

# import itertools
# num_frames = 8
# num_segments = 30
# scales = [i for i in range(num_frames, 1, -1)]

# outputs = torch.rand((3, 5))
# _, preds = torch.max(outputs, dim = 1)
# print(outputs)
# print("Max:", preds)


# print(scales)
# for scaleID in range(0, len(scales)):
#     print(scaleID, '----',scales[scaleID])

# print(x)

# generate the multiple relation scales
# x = list()
# for item in scales:
#     num_frames_relation = item
#     tmp = list(itertools.combinations(range(num_segments), num_frames_relation))
#     print('Scale %d'%item, len(tmp))
    # print(f"Loop {item}:", x)



# for scaleID in range(1, len(scales)):
#     print(x[scaleID])
#     idx_relations_randomsample = np.random.choice(len(x[scaleID]), 3, replace=False)
#     print('Scale %d'%scales[scaleID], idx_relations_randomsample)
#     for idx in idx_relations_randomsample:
#         print("Sample tupe:", x[scaleID][idx])
    
#     if scaleID >= 1:
#         break

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable


transform = transforms.Compose([      
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_frames(frames, num_frames=8):
    if len(frames) >= num_frames:
        return frames[::int(np.ceil(len(frames) / float(num_frames)))]
    else:
        raise (ValueError('Video must have at least {} frames'.format(num_frames)))

buffer_frames = []
def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m = torch.nn.Linear(20, 30).to(DEVICE)
    input = torch.randn(128, 20).to(DEVICE)
    output = m(input)
    print('output', output.size())
    exit()



    num_segs = 8
    N = 10
    i = 0
    input_bill = torch.rand((3,224,224))
    buffer_frames.append(input_bill)
    input_bill = torch.rand((3,224,224))
    buffer_frames.append(input_bill)
    data = buffer_frames[1:]
    print("Queue shape: ", torch.as_tensor(data).shape)

    # while(i <= N):
    #     input_bill = torch.rand((3,224,224))
    #     if len(buffer_frames) < num_segs:
    #         buffer_frames.append(input_bill)
    #     else:           
    #         lstt = torch.stack(buffer_frames)
    #         print("Queue shape: ", lstt.size())
    #         buffer_frames[:-1] = buffer_frames[1:]
    #         buffer_frames[-1] = input_bill
    #     i+=1

    
if __name__ == '__main__':

    main()
    

# input_bill = Variable(data.view(-1, 3, data.size(1), data.size(2)).unsqueeze(0).cuda(), volatile=True)

