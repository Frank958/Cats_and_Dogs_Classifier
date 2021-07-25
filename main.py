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

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=2)
ntrain = len(x_train)
nval = len(x_val)

batch_size = 32

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=1e-4), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True, )
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

history = model.fit(train_generator,
                    steps_per_epoch=ntrain // batch_size,
                    epochs=64,
                    validation_data=val_generator,
                    validation_steps=nval // batch_size)
