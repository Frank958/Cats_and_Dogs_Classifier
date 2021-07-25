import pandas as pd
import numpy as np
import cv2
import seaborn as sns
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import random
import gc

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

train_dir = 'C:\\Users\\phucp\\PycharmProjects\\Cats and Dogs Classifier\\train\\train'
test_dir = 'C:\\Users\\phucp\\PycharmProjects\\Cats and Dogs Classifier\\test1\\test1'

train_dogs = ['C:\\Users\\phucp\\PycharmProjects\\Cats and Dogs Classifier\\train\\train\\{}'.format(i) for i in
              os.listdir(train_dir) if 'dog' in i]
train_cats = ['C:\\Users\\phucp\\PycharmProjects\\Cats and Dogs Classifier\\train\\train\\{}'.format(i) for i in
              os.listdir(train_dir) if 'cat' in i]

test_imgs = ['C:\\Users\\phucp\\PycharmProjects\\Cats and Dogs Classifier\\test1\\test1\\{}'.format(i) for i in
             os.listdir(test_dir)]

train_imgs = train_dogs[:2000] + train_cats[:2000]
random.shuffle(train_imgs)

for ima in train_imgs[0:5]:
    img = mpimg.imread(ima)
    imgplot = plt.imshow(img)
    plt.savefig('1.png')
nrows = 150
ncolumns = 150
channels = 3


def read_and_process_image(list_of_images):
    x = []
    y = []

    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))

        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)

    return x, y


x, y = read_and_process_image(train_imgs)

plt.figure(figsize=(20, 10))
columns = 5
for i in range(columns):
    plt.subplot(2, columns, i + 1)
    plt.imshow(x[i])
    plt.savefig('2.png')

x = np.array(x)
y = np.array(y)

sns.displot(y)
plt.title('Labels for Cats and Dogs')
plt.savefig('3.png')

x_train, x_val, y_train, y_val = train_test_split(x,y, test_size=0.2,random_state=2)
print(x_val.shape)