import keras.models
import numpy
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

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img

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
random.shuffle(test_imgs)

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


model = models.load_model('model_keras.h5')

X_test, y_test = read_and_process_image(test_imgs[0:10])  # Y_test in this case will be empty
k = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1. / 255)

columns = 5

i = 0
text_labels = []
plt.figure(figsize=(30, 20))
for batch in test_datagen.flow(k, batch_size=1):
    pred = model.predict(batch)
    if pred > 0.5:
        text_labels.append('dog')
    else:
        text_labels.append('cat')
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title('This is a ' + text_labels[i])
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.savefig('6.png')
