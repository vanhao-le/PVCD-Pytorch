import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def total_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


# Creating a Custom Dataset for your files
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class TSNDataSet(data.Dataset):
    """
    Args:
    data_dir: file path of images for training / testing
    annotation_dir: file path of groundtruth e.g., train.csv / test.csv
    file_categories: file path of lables e.g., category.txt
    """
    def __init__(self, data_dir, annotation_dir, file_categories, num_segments=8, transform=None):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir        
        self.num_segments = num_segments        
        self.transform = transform
        self.classes = list(np.loadtxt(file_categories, dtype=np.str, delimiter=',')) #load category.txt
        self.video_list = list()
        anno_array = np.loadtxt(annotation_dir, dtype=np.str, delimiter=',')
        for anno in anno_array:            
            path = str(anno[0]).split('.')[0]
            label_name = anno[1]
            label = self.classes.index(label_name)
            data_path = os.path.join(data_dir, path)
            total_frames = len(os.listdir(data_path))
            # print(path, total_frames, label)
            self.video_list.append(VideoRecord([path, total_frames, label]))

    def _sample(self, num_total, num_segments):
        # sample = np.random.choice(range(num_total), size=num_segs, replace=None)
        # sample = np.sort(sample, axis=-1, kind='quicksort', order=None)
        sample = np.linspace(0, num_total-1, num_segments, endpoint=True, retstep=True, dtype=int)[0]       
        return sample

    def __len__(self):
        return len(self.video_list)

    # the __getitem__ method also allows you to turn your object into an iterable.

    def __getitem__(self, index):
        assert index < len(self.video_list)
        info = self.video_list[index]
        target = info.label
        total_frames = info.total_frames
        data_path = os.path.join(self.data_dir, info.path)
        image_path_list = os.listdir(data_path)
        image_list = list()
        sample = self._sample(total_frames, self.num_segments)
        for i in sample:
            img_path = os.path.join(data_path, image_path_list[i])
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            image_list.append(img)
        image = torch.stack(image_list)
        return image, target

# Creating a Custom Dataset for your files
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

class VCDBDataset(data.Dataset):
    """
    Args:
    data_dir: file path of images for training / validation
    annotation_dir: file path of groundtruth e.g., train.csv / val.csv
    file_categories: file path of lables e.g., category.csv
    """
    def __init__(self, data_dir, annotation_dir, file_categories, num_segments=8, transform=None):
        self.data_dir = data_dir
        self.annotation_dir = annotation_dir        
        self.num_segments = num_segments        
        self.transform = transform
        self.classes = pd.read_csv(file_categories)
        self.video_list = list()
        df_annotation = pd.read_csv(annotation_dir)
        for item in df_annotation.itertuples():            
            path = item.video_name.split('.')[0]
            label_name = item.classIDx
            label = label_name
            data_path = os.path.join(data_dir, path)
            total_frames = len(os.listdir(data_path))
            # print(path, total_frames, label)
            self.video_list.append(VideoRecord([path, total_frames, label]))

    def _sample(self, num_total, num_segments):
        # sample = np.random.choice(range(num_total), size=num_segs, replace=None)
        # sample = np.sort(sample, axis=-1, kind='quicksort', order=None)
        sample = np.linspace(0, num_total-1, num_segments, endpoint=True, retstep=True, dtype=int)[0]       
        return sample

    def __len__(self):
        return len(self.video_list)

    # the __getitem__ method also allows you to turn your object into an iterable.

    def __getitem__(self, index):
        assert index < len(self.video_list)
        info = self.video_list[index]
        target = info.label
        total_frames = info.total_frames
        data_path = os.path.join(self.data_dir, info.path)
        image_path_list = os.listdir(data_path)
        image_list = list()
        sample = self._sample(total_frames, self.num_segments)
        for i in sample:
            img_path = os.path.join(data_path, image_path_list[i])
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            image_list.append(img)
        image = torch.stack(image_list)
        return image, target

