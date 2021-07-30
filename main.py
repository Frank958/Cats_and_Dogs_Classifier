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

from keras.applications import InceptionResNetV2

conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

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
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False

model.compile(loss='binary_crossentropy', optimizer=optimizers.rmsprop(lr=2e-5), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size)

history = model.fit_generator(train_generator,
                              steps_per_epoch=ntrain // batch_size,
                              epochs=20,
                              validation_data=val_generator,
                              validation_steps=nval // batch_size)

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
# Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('4.png')

X_test, y_test = read_and_process_image(test_imgs[0:10])  # Y_test in this case will be empty
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1. / 255)

i = 0
text_labels = []
plt.figure(figsize=(30, 20))
for batch in test_datagen.flow(x, batch_size=1):
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
plt.savefig('5.png')
