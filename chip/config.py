# ========================= Common Configs ==========================
'''
ucf5 dataset
'''

# dataset_root = "data\\ucf5"
# file_categories = r"data\ucf5\category.txt"
# file_imglist_train = r"data\ucf5\train"
# file_imglist_test = r"data\ucf5\test"
# filename_imglist_train = r"data\ucf5\train.csv"
# filename_imglist_val = r"data\ucf5\test.csv"

'''
VCDB dataset
'''

dataset_root = "data\\vcdb"
file_categories = r"data\vcdb\category.csv"
file_imglist_train = r"data\vcdb\train"
file_imglist_test = r"data\vcdb\val"
filename_imglist_train = r"data\vcdb\train.csv"
filename_imglist_val = r"data\vcdb\val.csv"

prefix = '{:06d}.jpg'

# ========================= Model Configs ==========================
num_segments = 30
d_frames = 8
k_random = 3
num_classes = 28
image_size = 224

# ========================= Learning Configs ==========================
batch_size = 4
num_workers = 2
learning_rate = 1e-3
momentum = 0.9
weight_decay = 5e-4
step_size = 10
num_epochs = 50
dropout = 0.6
img_feature_dim = 2048 #the feature dimension for each frame

