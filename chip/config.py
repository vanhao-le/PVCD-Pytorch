# ========================= Common Configs ==========================
dataset_root = "data\\ucf5"
file_categories = r"data\ucf5\category.txt"
file_imglist_train = r"data\ucf5\train"
file_imglist_test = r"data\ucf5\test"
filename_imglist_train = r"data\ucf5\train.csv"
filename_imglist_val = r"data\ucf5\test.csv"

prefix = '{:06d}.jpg'
store_name = "models"

# ========================= Model Configs ==========================
num_segments = 25
d_frames = 8
k_random = 3
num_classes = 5

# ========================= Learning Configs ==========================
batch_size = 5
num_workers = 2
learning_rate = 1e-3
momentum = 0.9
weight_decay = 5e-4
step_size = 20
num_epochs = 10
dropout = 0.8
img_feature_dim = 2048 #the feature dimension for each frame

