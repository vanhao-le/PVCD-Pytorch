import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

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


    # def _load_image(self, directory, idx):

    #     try:
    #         return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
    #     except Exception:
    #         print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
    #         return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]        

    # def _parse_list(self):
    #     # check the frame number is large >3:
    #     # usualy it is [video_id, num_frames, class_idx]
    #     tmp = [x.strip().split(' ') for x in open(self.list_file)]
    #     tmp = [item for item in tmp if int(item[1])>=3]
    #     self.video_list = [VideoRecord(item) for item in tmp]
    #     print('video number:%d'%(len(self.video_list)))

    # def _sample_indices(self, record):
    #     """
    #     :param record: VideoRecord
    #     :return: list
    #     """

    #     average_duration = record.num_frames // self.num_segments
    #     if average_duration > 0:
    #         offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
    #     elif record.num_frames > self.num_segments:
    #         offsets = np.sort(randint(record.num_frames, size=self.num_segments))
    #     else:
    #         offsets = np.zeros((self.num_segments,))

    #     # print("Offsets: ", offsets)
    #     return offsets + 1

    

    # def _get_val_indices(self, record):
    #     if record.num_frames > self.num_segments:
    #         tick = record.num_frames / float(self.num_segments)
    #         offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
    #     else:
    #         offsets = np.zeros((self.num_segments,))
    #     return offsets + 1

    # def _get_test_indices(self, record):
    #     tick = record.num_frames  / float(self.num_segments)
    #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
    #     return offsets + 1

    # # the __getitem__ method also allows you to turn your object into an iterable.
    # def __getitem__(self, index):
    #     record = self.video_list[index]
    #     # check this is a legit video folder
    #     while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
    #         print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
    #         index = np.random.randint(len(self.video_list))
    #         record = self.video_list[index]

    #     if not self.test_mode:
    #         segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
    #     else:
    #         segment_indices = self._get_test_indices(record)

    #     return self.get(record, segment_indices)

    # def get(self, record, indices):

    #     images = list()
    #     for seg_ind in indices:
    #         p = int(seg_ind)           
    #         seg_imgs = self._load_image(record.path, p)
    #         images.extend(seg_imgs)
    #         if p < record.num_frames:
    #             p += 1

    #     process_data = self.transform(images)
    #     return process_data, record.label

   