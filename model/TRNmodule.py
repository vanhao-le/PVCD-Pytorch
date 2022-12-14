import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# Concatenates the given sequence of seq tensors in the given dimension. 
# All tensors must either have the same shape (except in the concatenating dimension) or be empty.
'''
x = torch.randn(2, 3)
y = torch.cat((x, x, x), 1)
y shapes [2, 9]
'''
class MergeFrame(nn.Module):
    def __init__(self):
        super(MergeFrame, self).__init__()        
    
    def forward(self, x):
        if(len(x) == 1):
            return x[0]
        return torch.cat(x, dim=1)

class RelationModule(nn.Module):
    # this is the naive implementation of the n-frame relation module, as d_frames == d_frames_relation
    def __init__(self, img_feature_dim, d_frames, num_class):
        super(RelationModule, self).__init__()
        self.d_frames = d_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()
    
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 256    # This part is chosen by the author.
        classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.d_frames * self.img_feature_dim, num_bottleneck),
            nn.ReLU(),
            nn.Linear(num_bottleneck, self.num_class),
            )
        return classifier
    
    def forward(self, input):
        input = input.view(input.size(0), self.d_frames * self.img_feature_dim)
        input = self.classifier(input)
        return input


class RelationModuleMultiScale(nn.Module):
    '''
    A video has M frames
    A segments has N frames
    A relation has d \in [2, N] combinated tuples such as tuples 2-frames, 3-frames, ... n-frames
    '''
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, d_frames, num_segments, num_class):
        super(RelationModuleMultiScale, self).__init__()

        # subsample_num:  The parameter k mentioned in the paper.
        self.subsample_num = 3   # how many relations selected to sum up
        self.num_segments = num_segments
        self.img_feature_dim = img_feature_dim
        # range(d_frames, 1, -1) where d_frames = 8 -> [8, 7, 6, ..., 2]
        '''
        scale means how many kinds of combination
        for example: if d_frames = 3 then we have 2-frames and 3-frames relattion types
        if d_frame = 5 then we have 2-frames, 3-frames, 4-frames, 5-frames relation types
        '''
        self.scales = [i for i in range(d_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_segments, scale)
            self.relations_scales.append(relations_scale)
            # subsample_scales: how many samples of the relation to select in each forward pass
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) 

        self.num_class = num_class
        self.d_frames = d_frames
        num_bottleneck = 256
        # ModuleList holds submodules in a list.
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Dropout(p=0.6), # this is the newly added thing
                        nn.Linear(num_bottleneck, self.num_class),                        
                        )

            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):        
        '''
        self.relations_scales dimension
        dim=0 means the kind of relations 
        For example: self.relations_scales[0] = 8-frame relations, self.relations_scales[1] = 7-frames relations
        self.scales \in [d_frames, 2] -> self.scales[0] = 8        
        '''
        # the first one is the largest scale

        # input shape: [batch_size, num_segments, img_feature_dim]
        # print("Input shape:", input.shape)
        act_all = input[:, self.relations_scales[0][0] , :]
        
        # act_all shape: [batch_size, d_frames, img_feature_dim]        
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)        
        act_all = self.fc_fusion_scales[0](act_all)
        
             
        #  for another scales
        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            print('Scales %d:'%self.scales[scaleID], idx_relations_randomsample)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation        
            
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        
        '''
        itertools.combinations(iterable, r)
        Return r length subsequences of elements from the input iterable.
        Note: The combination tuples are emitted in lexicographic ordering according to the order of the input iterable.
        So, if the input iterable is sorted, the output tuples will be produced in sorted order.
        Elements are treated as unique based on their position, not on their value. 
        So if the input elements are unique, there will be no repeated values in each combination.
        Ex: combinations(range(4), 3) --> 012 013 023 123
        The number of items returned is n! / r! / (n-r)!  when 0 <= r <= n
        '''       
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))


'''
Code testing generation of the combination with different scales

import itertools
num_frames = 8
scales = [i for i in range(num_frames, 1, -1)]
# print(x)
for item in scales:
    num_frames_relation = item
    x = list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))
    print(f"Loop {item}:", x)
'''