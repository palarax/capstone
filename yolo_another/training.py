import os
import argparse
import yaml

import cv2  # image processing
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plot 

from mymodel.PreTrainedInceptionResNet import PreTrainedInceptionResNet
from keras.optimizers import RMSprop

config = yaml.safe_load(open("configuration/config.yaml"))


def prepareModel(noClasses, final_act, learning_rate, loss_fun, loss_met):
    """
    Prepare model through transfer learning
    noClasses: Number of Classes
    final_act: Final activation in FC layer
    learning_rate: learning rate of the model
    loss_fun: loss function used in propogation
    loss_met: metrics used
    """
    print("[INFO] Prepare model...")
    # get our modified model
    model = PreTrainedInceptionResNet.prepareModel(noClasses, final_act)
    # optimizer
    opt = RMSprop(lr=learning_rate)
    model.compile(loss=loss_fun, optimizer=opt, metrics=loss_met)
    return model

def prepareData():
    """
    Prepare data for training
    """
    print("[INFO] preparing dataset")

    return NotImplementedError


def plot():
    """Plot the training loss and accuracy
    """
    print("[INFO] plotting loss and accuracy")
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(args["plot"])

if __name__ == "__main__":
    model = prepareModel(len(config["Global"]["CLASSES"]),
                         config["MODEL"]["FINAL_ACTIVATION"],
                         float(config["MODEL"]["LEARNING_RATE"]),
                         config["MODEL"]["LOSS_FUNCTION"],
                         config["MODEL"]["LOSS_METRICS"])
    # model.summary()
    # preprare data
    # train network
    # train the network
    print("[INFO] training network...")
    # H = model.fit_generator(
    #     aug.flow(trainX, trainY, batch_size=BS),
    #     validation_data=(testX, testY),
    #     steps_per_epoch=len(trainX) // BS,
    #     epochs=EPOCHS, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    # model.save(config["Global"]["MODEL_PATH"])
    # plot results