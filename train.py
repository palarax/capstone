#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import logging

# KERAS
import tensorflow as tf
import keras.backend as K
import h5py  # for saving model
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# YOLO model
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data

from keras.utils import plot_model  # plot model


# Initialize GPU
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Initialize logger
logger = None


def setup_logger(log, filename, loglevel, format, dtfmt):
    """Configure logger
    """
    logger = logging.getLogger(log)
    # Set Log Level and format
    numeric_level = getattr(logging, loglevel.upper(), None)
    logger.setLevel(level=numeric_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level=numeric_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # create file handler
    fh = logging.FileHandler(filename)
    fh.setLevel(level=numeric_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # y_true = [Input(shape=(416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 80+5)) for l in range(3)]
    y_true = [Input(shape=(h//{0: 32, 1: 16, 2: 8}[l], w//{0: 32, 1: 16, 2: 8}[
                    l], num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('[INFO] Create YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        print('[INFO] Load weights {}.'.format(weights_path))
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('[INFO] Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)
    # print('[INFO] model_body.input: ', model_body.input)
    # print('[INFO] model.input: ', model.input)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0: 32, 1: 16}[l], w//{0: 32, 1: 16}[l],
                           num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(
        num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(
                num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def _main_(args):

    config_path = args.conf
    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Setup logger
    ###############################
    # TODO: replace prints with logger
    setup_logger(
        config["logger"]["logger"],
        config["logger"]["filename"],
        config["logger"]["level"],
        config["logger"]["format"],
        config["logger"]["datefmt"])

    log_dir = config["train"]["tensorboard_dir"]
    class_names = config["model"]["labels"]
    num_classes = len(class_names)
    anchors = np.array(config["model"]["anchors"]).reshape(-1, 2)

    ###############################
    #   Parse the annotations
    ###############################
    print("[INFO] Parsing annotations")

    ###############################
    #   Create the generators
    ###############################
    print("[INFO] Creating Training Generators")

    print("[INFO] Creating Validation Generators")

    ###############################
    #   Create the model
    ###############################
    print("[INFO] Creating Model")

    # TODO: figure out what this does
    input_shape = (416, 416)  # multiple of 32, hw

    is_tiny_version = len(anchors) == 6  # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2, weights_path=config["model"]["weights"])  # 'model_data/tiny_yolo_weights.h5'
    else:
        model = create_model(input_shape, anchors, num_classes,
                             freeze_body=2, weights_path=config["model"]["weights"])  # make sure you know what you freeze
    # architecture + weights + optimizer state
    # allows to resume from where you left off
    # model.save('yolo_model_retrain.hdf5')  # creates a HDF5 file 'my_model.h5'
    #model.summary()

    ###############################
    #   Kick off the training
    ###############################
    print("[INFO] Creatinng Callbacks")
    print("[INFO] Start Training")


if __name__ == '__main__':
    # TODO: add python logger
    argparser = argparse.ArgumentParser(
        description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
