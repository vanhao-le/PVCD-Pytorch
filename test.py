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


import itertools
num_frames = 8
num_segments = 30
scales = [i for i in range(num_frames, 1, -1)]

outputs = torch.rand((3, 5))
_, preds = torch.max(outputs, dim = 1)
print(outputs)
print("Max:", preds)


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
