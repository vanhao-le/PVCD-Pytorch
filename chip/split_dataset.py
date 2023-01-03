'''
This code used to split a dataset dedicated to the classification problem into three parts as train / val / test.

Train dataset: Set of data used for learning in order to fit the parameters to the machine learning model.
Valid dataset: Set of data used to provide an unbiased evaluation of a model fitted on the training dataset while tuning model hyperparameters.
In addition, it also plays a role in other forms of model preparation, such as feature selection, threshold cut-off selection.
Test dataset: Set of data used to provide an unbiased evaluation of a final model fitted on the training dataset.

Agrs: The csv input file formats as [video_name, class_name]

Return:
Three files train.csv / val.csv / test.csv

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

groundtruth_file = r'chip\data.csv'
df = pd.read_csv(groundtruth_file)

# suffle data before splitting
df = shuffle(df)

# Let's say we want to split the data in 80/10/10 for train/valid/test dataset
train_size=0.8

data = df.drop(columns = ['class_name']).copy()
lables = df['class_name']

# In the first step we will split the data in training and remaining dataset
x_train, x_remain, y_train, y_remain = train_test_split(data, lables, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)

test_size = 0.5
x_valid, x_test, y_valid, y_test = train_test_split(x_remain, y_remain, test_size = 0.5)

# print(x_train.shape), print(y_train.shape)
# print(X_valid.shape), print(y_valid.shape)
# print(X_test.shape), print(y_test.shape)

'''
Concatenate pandas objects along a particular axis.
axis{0/'index', 1/'columns'}, default 0

Example:
s1 = pd.Series(['a', 'b'])
s2 = pd.Series(['0', '1'])
s = pd.concat([s1, s2], axis= 1)

Result:
   0  1
0  a  0
1  b  1

'''

train_ds = pd.concat([x_train, y_train], axis=1)
val_ds = pd.concat([x_valid, y_valid], axis=1)
test_ds = pd.concat([x_test, y_test], axis=1)

train_ds.to_csv(r'chip\train.csv', columns=['video_name', 'class_name'], index=False, header=True, encoding="utf-8-sig")
val_ds.to_csv(r'chip\val.csv', columns=['video_name', 'class_name'], index=False, header=True, encoding="utf-8-sig")
test_ds.to_csv(r'chip\test.csv', columns=['video_name', 'class_name'], index=False, header=True, encoding="utf-8-sig")

