#! /usr/bin/env python

import argparse
import numpy as np
import json
import logging

# KERAS
import tensorflow as tf
import keras.backend as K
# import h5py  # for saving model
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler

# YOLO model
from yolo3.model import yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import data_generator_wrapper


# Initialize GPU
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


class PolynomialDecay():
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay

        # return the new learning rate
        return float(alpha)

def lr_schedule(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 10:
        return 0.0001
    else:
        return 0.00001
		

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
    # annotations = config["train"]["annotations"]

    ##################################
    #   Create validation and training
    ##################################
    print("[INFO] Seperating Training and Validation")

    # <dir>/img x1 y1 x2 y2
    # images_dir = './dataset/images/'
    val_annot = './dataset/yolo_val.txt'
    train_annot = './dataset/yolo_train.txt'

    with open(val_annot) as f:
        val = f.readlines()

    with open(train_annot) as f:
        train = f.readlines()

    ###############################
    #   Create the model
    ###############################
    print("[INFO] Creating Model")

    # input_shape = (416, 416)  # multiple of 32, hw
    input_shape = (608, 608)

    model = create_model(input_shape, anchors, num_classes,
                                # freeze_body=2, weights_path="model_stages/004/trained_model_stage_1.h5")
                             freeze_body=2, weights_path=config["model"]["weights"])  # make sure you know what you freeze

    ###############################
    #   Monitoring
    ###############################
    print("[INFO] Creatinng Callbacks")
    # For the actual monitoring
    tensorboard = TensorBoard(log_dir=config["train"]["tensorboard_dir"])
    checkpoint = ModelCheckpoint(config["train"]["model_stages"] + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=False, save_best_only=True, period=5)
    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.01, patience=2, verbose=1)

    # reduce_lr = ReduceLROnPlateau(
    #     monitor='val_loss', factor=0.001, patience=2, verbose=1)

    schedule = PolynomialDecay(maxEpochs=config["train"]["epochs"], initAlpha=1e-3, power=2)  # TODO
    # # schedule = lr_schedule
    learning_rate_scheduler = LearningRateScheduler(schedule=schedule, verbose=1)
    # optim = Adam(lr=1e-3)
    optim = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optim = SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=False)

    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    ###############################
    #   Kick off the training
    ###############################
    print("[INFO] Start Training")
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.

    if True:
        model.compile(optimizer=optim, loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred},
            metrics=['mean_squared_error'])

        batch_size = config["train"]["batch_size"]

        print('[INFO] Train on {} samples, val on {} samples, with batch size {}.'.format(
            len(train), len(val), batch_size))
        model.fit_generator(data_generator_wrapper(train, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, len(train)//batch_size),
                            validation_data=data_generator_wrapper(
                                val, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, len(val)//batch_size),
                            initial_epoch=0,
                            epochs=config["train"]["epochs"],
                            callbacks=[tensorboard,learning_rate_scheduler, checkpoint])
        # model.save_weights(log_dir + 'trained_weights_stage_1.h5')
        model.save_weights(config["train"]["model_stages"] +
                           'trained_model_stage_1.h5')

    schedule = PolynomialDecay(maxEpochs=config["train"]["epochs_unfrozen"], initAlpha=1e-4, power=2)
    learning_rate_scheduler = LearningRateScheduler(schedule=schedule, verbose=1)
    optim = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # optim = SGD(lr=1e-5, momentum=0.9, decay=0.0, nesterov=False)
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        model.compile(optimizer=optim, loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred},
            metrics=['mean_squared_error'])

        print('[INFO] Unfreeze all of the layers.')

        batch_size = 1  # note that more GPU memory is required after unfreezing the body
        print('[INFO] Train on {} samples, val on {} samples, with batch size {}.'.format(
            len(train), len(val), batch_size))
        model.fit_generator(data_generator_wrapper(train, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, len(train)//batch_size),
                            validation_data=data_generator_wrapper(
                                val, batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, len(val)//batch_size),
                            initial_epoch=config["train"]["epochs"],
                            epochs=config["train"]["epochs_unfrozen"],
                            callbacks=[tensorboard, checkpoint, learning_rate_scheduler, early_stopping])
                            # callbacks=[tensorboard, checkpoint, reduce_lr, early_stopping])
        model.save_weights(config["train"]["model_stages"] +
                           'trained_model_stage_2.h5')


if __name__ == '__main__':
    # TODO: add python logger
    # TODO: https://www.youtube.com/watch?v=BqgTU7_cBnk
    # https://github.com/bing0037/keras-yolo3
    argparser = argparse.ArgumentParser(
        description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file')

    args = argparser.parse_args()
    _main_(args)
