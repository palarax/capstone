import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

config = yaml.safe_load(open("config.yaml"))


def defineNetwork(pooling_average, weight_path, num_classes, dense_layer_activation):
    """
     https://www.kaggle.com/suniliitb96/tutorial-keras-transfer-learning-with-resnet50
    """
    model = Sequential()

    # 1st layer as the lumpsum weights from resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    # NOTE that this layer will be set below as NOT TRAINABLE, i.e., use it as is
    model.add(ResNet50(include_top=False,
                       pooling=pooling_average, weights=weight_path))

    # 2nd layer as Dense for 2-class classification, i.e., dog or cat using SoftMax activation
    model.add(Dense(num_classes, activation=dense_layer_activation))

    # Say not to train first layer (ResNet) model as it is already trained
    model.layers[0].trainable = False
    return model


if __name__ == "__main__":

    model = defineNetwork(config["Global"]["RESNET50_POOLING_AVERAGE"],
                          config["Global"]["WEIGHT_PATH"], config["Global"]["NUM_CLASSES"],
                          config["Global"]["DENSE_LAYER_ACTIVATION"]
                          )
    model.summary()
