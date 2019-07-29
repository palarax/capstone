import os
import argparse
import yaml

import cv2 # image processing
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Keras imports
from keras.models import  Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

# InceptionV3 model imports
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16


config = yaml.safe_load(open("configuration/config.yaml"))


def defineModel(noClasses):
    """
    TODO
    """
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)
    # and a logistic layer with noClasses
    predictions = Dense(noClasses, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

# TODO
if __name__ == "__main__":
    # https://www.kaggle.com/venuraja79/using-transfer-learning-with-keras
    # https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/
    defineModel(5)
