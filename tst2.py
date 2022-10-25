import keras.layers
import numpy as np
import matplotlib as plt
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import cv2
import helpers
import random
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hyp
from sklearn.model_selection import train_test_split
import os
import shutil

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

hparams = [{'activation': 'tanh'}, {'activation': 'relu'}, {'activation': 'linear'}, {'activation': "elu"}, {'activation': 'selu'}, {'activation': 'exponential'}]

tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

train_dir = '/home/gleb/sign/training'
val_dir = '/home/gleb/sign/val'
test_dir = '/home/gleb/sign/test'
files_1 = []
files_2 = []
# os.chdir('/home/mike/sign/val')

def split_move(test_dir, split=0.2):
    for r, dirs, files in os.walk(test_dir):
        print(r)
        if dirs != []:
            for dir in dirs:
                try:
                    os.mkdir(dir)
                except:
                    pass
        else:
            if files != []:
                test_f, val_f = train_test_split(files, test_size=split)
            else:
                continue
            for file in test_f:
                shutil.move(r + '/' + file,
                            r[:len('/home/gleb/sign')] + '/val' + r[len('/home/gleb/sign/test'):] + '/' + file)



img_w, img_h = 64, 64
input_shape = (img_w, img_h, 3)
epochs = 12
batch_s = 64
nb_train_s = 13220
nb_val_s = 201
nb_test_s = 54

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(train_dir,
                                              target_size=(img_w, img_h),
                                              batch_size=batch_s,
                                              class_mode='categorical')

val_generator = datagen.flow_from_directory(val_dir,
                                            target_size=(img_w, img_h),
                                            batch_size=batch_s,
                                            class_mode='categorical')

test_generator = datagen.flow_from_directory(test_dir,
                                             target_size=(img_w, img_h),
                                             batch_size=1,
                                             class_mode='categorical')

for hp in hparams:
    i = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', **hp)(i)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', **hp)(x)
    # x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2, 2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, **hp)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    o = tf.keras.layers.Dense(9, activation='softmax')(x)
    model = Model(inputs=i, outputs=o)
    model.summary()

    if hp == {'activation': tf.keras.activations.hard_sigmoid}:
        name = 'hard_sigm'
    else:
        name = hp['activation']
    callback = [TensorBoard(log_dir=f"/home/gleb/PycharmProjects/log/{name}",update_freq=5)]
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=["accuracy"])
    model.fit(train_generator,
              epochs=epochs,
              validation_data=val_generator,
              callbacks=callback)
    model.evaluate(test_generator,callbacks=callback)


