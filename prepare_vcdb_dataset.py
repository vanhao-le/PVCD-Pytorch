import pandas as pd
import os
from sklearn.model_selection import train_test_split
import shutil
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from numpy.random import randint


'''
Load videos and generate the data.csv as format [video_name,classIDx]

Argus:
The root video path

Returns:
data.csv file


'''

def generate_data(video_path):

    # load category file
    label_file = r'data\vcdb\category.csv'
    df = pd.read_csv(label_file)

    data = list()

    for item in df.itertuples():
        sub_path = item.class_name
        IDx = item.classIDx

        vid_path = os.path.join(video_path, sub_path)        
        # Note: This would include all the files and directories
        directory_list = os.listdir(vid_path)
        for file in directory_list:
            case = {'video_name':file, 'classIDx':IDx}
            data.append(case)        
    
    df = pd.DataFrame(data)
    output_file = r'data\vcdb\data.csv'
    df.to_csv(output_file, columns=['video_name', 'classIDx'], header=True, index=False)

def split_train_val_test(root_path, data_file):
    groundtruth_file = os.path.join(root_path, data_file)
    df = pd.read_csv(groundtruth_file)
    train_size = 0.7

    data = df.drop(columns = ['classIDx']).copy()
    lables = df['classIDx']
    x_train, x_remain, y_train, y_remain = train_test_split(data, lables, train_size= train_size)
    test_size = 0.5
    x_valid, x_test, y_valid, y_test = train_test_split(x_remain, y_remain, test_size = test_size)
    train_ds = pd.concat([x_train, y_train], axis=1)
    val_ds = pd.concat([x_valid, y_valid], axis=1)
    test_ds = pd.concat([x_test, y_test], axis=1)

    train_file = os.path.join(root_path, 'train.csv')
    val_file = os.path.join(root_path, 'val.csv')
    test_file = os.path.join(root_path, 'test.csv')

    train_ds.to_csv(train_file, columns=['video_name', 'classIDx'], index=False, header=True, encoding="utf-8-sig")
    val_ds.to_csv(val_file, columns=['video_name', 'classIDx'], index=False, header=True, encoding="utf-8-sig")
    test_ds.to_csv(test_file, columns=['video_name', 'classIDx'], index=False, header=True, encoding="utf-8-sig")


def copy_video():

    train_file = r'data\vcdb\train.csv'
    val_file =  r'data\vcdb\val.csv'
    test_file =  r'data\vcdb\test.csv'

    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)
    df_test = pd.read_csv(test_file)

    original_path = r'D:\pvcd_core' 
    train_path = r'D:\vcdb_split\train'
    val_path = r'D:\vcdb_split\val'
    test_path = r'D:\vcdb_split\test'

    for item in df_train.itertuples():
        video_file = item.video_name
        old_path = os.path.join(original_path, video_file)
        new_path = os.path.join(train_path, video_file)
        shutil.copyfile(old_path, new_path)
    
    for item in df_val.itertuples():
        video_file = item.video_name
        old_path = os.path.join(original_path, video_file)
        new_path = os.path.join(val_path, video_file)
        shutil.copyfile(old_path, new_path)
    
    for item in df_test.itertuples():
        video_file = item.video_name
        old_path = os.path.join(original_path, video_file)
        new_path = os.path.join(test_path, video_file)
        shutil.copyfile(old_path, new_path)

def main():
    '''
    step 1: generate the all of video files and correspoding classIDxs
    '''
    root_video_path = r'D:\video\VCDB\pvcd_core'
    # generate_data(root_video_path)

    '''
    split data for train / val / test parts
    '''
    root_path = r'data\vcdb'
    data_file = r'data.csv'

    # split_train_val_test(root_path, data_file)

    '''
    Copy videos to  train / val / test folders
    '''

    # copy_video()

    print()

if __name__ == '__main__':
    main()




