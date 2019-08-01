#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import logging

# KERAS
import tensorflow as tf
import keras.backend as K
# import h5py  # for saving model
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# YOLO model
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data, data_generator_wrapper

from keras.utils import plot_model  # plot model


# Initialize GPU
from keras.backend.tensorflow_backend import set_session
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#config.gpu_options.allow_growth = True
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

    class_names = config["model"]["labels"]
    num_classes = len(class_names)
    anchors = np.array(config["model"]["anchors"]).reshape(-1, 2)

    ###############################
    #   Parse the annotations
    ###############################
    print("[INFO] Parsing annotations")
    annotations = config["train"]["annotations"]

    ##################################
    #   Create validation and training
    ##################################
    print("[INFO] Seperating Training and Validation")

    val_split = 0.1
    with open(annotations) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

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

    # Literally outputs an image of a model
    #plot_model(model, to_file='output/retrained_model.png', show_shapes = True)

    ###############################
    #   Monitoring
    ###############################
    print("[INFO] Creatinng Callbacks")
    tensorboard = TensorBoard(log_dir=config["train"]["tensorboard_dir"]) # For the actual monitoring
    checkpoint = ModelCheckpoint(config["train"]["model_stages"] + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=False, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    ###############################
    #   Kick off the training
    ###############################
    print("[INFO] Start Training")
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3),
                      loss='mean_squared_error', metrics=['accuracy'])
        batch_size = config["train"]["batch_size"]
        print('[INFO] Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(
                                lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=config["train"]["epochs"],
                            initial_epoch=0,
                            callbacks=[tensorboard, checkpoint])
        # model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        model.save(config["train"]["model_stages"] +
                   'trained_model_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-4),
                      loss='mean_squared_error', metrics=['accuracy'])
        print('[INFO] Unfreeze all of the layers.')

        batch_size = 1  # note that more GPU memory is required after unfreezing the body
        print('[INFO] Train on {} samples, val on {} samples, with batch size {}.'.format(
            num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(
                                lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping])
        model.save(config["train"]["model_stages"] + 'train_model_final.h5')

    derived_model = Model(model.input[0], [
                          model.layers[249].output, model.layers[250].output, model.layers[251].output])
    
    derived_model.save(config["train"]["model_stagers"])
    plot_model(derived_model, to_file='output/derived_model.png',
               show_shapes=True)



if __name__ == '__main__':
    # TODO: add python logger
    # TODO: https://www.youtube.com/watch?v=BqgTU7_cBnk
    argparser = argparse.ArgumentParser(
        description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
