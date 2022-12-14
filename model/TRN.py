import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .resnet import resnet50
from .TRNmodule import MergeFrame, RelationModule, RelationModuleMultiScale

# Resnet: C:\Users\le/.cache\torch\hub\checkpoints\resnet50-19c8e357.pth
class TRN(nn.Module):
    def __init__(self, img_feature_dim, d_frames, num_segments, num_class):
        super(TRN, self).__init__()

        self.num_class = num_class
        self.d_frames = d_frames
        self.num_segments = num_segments
        self.img_feature_dim = img_feature_dim
        self.backbone = resnet50(pretrained=True)
        self.merge = MergeFrame()
        self.TRN = RelationModule(img_feature_dim, d_frames, num_class)
    
    def forward(self, x):
        # x shape [batch_size, d_frames, num_channels, width, height]
        # torch.Size([5, 8, 3, 224, 224]) -> total_frames = 8
        # print("Input shape", x.shape)        
        total_frame = x.shape[1]        
        # Chunk: Attempts to split a tensor into the specified number of chunks. 
        # Each chunk is a view of the input tensor.
        # torch.chunk(input, chunks, dim=0)
        x_frame = torch.chunk(x, total_frame, 1)
        # pick up 'd_frames' from range (0, total_frames)
        sample = np.random.choice(range(total_frame), size=self.d_frames, replace=None)
        sample = np.sort(sample, axis=-1, kind='quicksort', order=None)
        x_feature = list()
        for i in sample:
            '''
            squeeze: Returns a tensor with all the dimensions of input of size 1 removed.
            torch.zeros(2, 1, 2, 1, 2) -> torch.Size([2, 1, 2, 1, 2])
            y = torch.squeeze(x, 1) -> torch.Size([2, 2, 1, 2])
            y = torch.squeeze(x, 0) -> torch.Size([2, 1, 2, 1, 2])
            '''
            
            frame = torch.squeeze(x_frame[i], 1)

            '''
            unsqueeze: Returns a new tensor with a dimension of size one inserted at the specified position.
            x = torch.tensor([1, 2, 3, 4])
            torch.unsqueeze(x, 0) -> tensor([[ 1,  2,  3,  4]])
            torch.unsqueeze(x, 1) -> tensor([[ 1], [ 2], [ 3], [ 4]])
            '''

            feature_extracted =  self.backbone(frame)
            x_feature.append(torch.unsqueeze(feature_extracted, 1))

            # print("feature shape: ", feature_extracted.shape) #2048
            

        x = self.merge(x_feature)
        # print("Input shape: ", x.shape) #torch.Size([5, 2, 2048]) 
        x = self.TRN(x)
        return x


class MultiScaleTRN(nn.Module):
    def __init__(self, img_feature_dim, d_frames, num_segs, num_class):
        super(MultiScaleTRN, self).__init__()
        self.num_class = num_class
        self.d_frames = d_frames
        self.img_feature_dim = img_feature_dim
        self.backbone = resnet50(pretrained=True)
        self.merge = MergeFrame()
        self.TRN = RelationModuleMultiScale(img_feature_dim, d_frames, num_segs, num_class)        
    
    def forward(self, x):
        total_frame = x.shape[1]
        x_frame = torch.chunk(x, total_frame, 1)
        x_feature = list()
        for i in range(0, total_frame):
            x_feature.append(torch.unsqueeze(self.backbone(torch.squeeze(x_frame[i], 1)), 1))
        x = self.merge(x_feature)
        x = self.TRN(x)
        return x



def test():
    batch_size = 5
    d_frames = 8
    num_segments = 30
    num_class = 5
    img_feature_dim = 2048
    input_var = Variable(torch.randn(batch_size, num_segments, 3, 224, 224))   
    # model = TRN(img_feature_dim, d_frames, num_segments, num_class)
    model = MultiScaleTRN(img_feature_dim, d_frames, num_segments, num_class)
    output = model(input_var)
    print(output)
    print(output.shape)



# test()
