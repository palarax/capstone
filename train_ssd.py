
# %%
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from decimal import Decimal

# Deep learning
import keras
import tensorflow as tf
# monitoring callbacks
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
# from keras import backend as K
from keras.models import Model
from keras.preprocessing import image

# Custom classes and libraries
# from model.ssd300MobileNet import SSD
from model.ssd300VGG16 import SSD
from utils.ssd_training import MultiboxLoss
from utils.ssd_utils import BBoxUtility
from utils.generator import Generator
from utils.utils import get_annotations, isWeightsChanged, get_weights_layers

# VALIDATION
from scipy.misc.pilutil import imread
from keras.applications.imagenet_utils import preprocess_input


# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (8, 8)
# plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)
#######################################################################
#   Setup GPU
#######################################################################
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
# SESSION = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# set_session(SESSION)
# K.clear_session() # Clear previous models from memory.
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))

# %%
# some constants
NUM_CLASSES = 5  # positive clases TODO: 0 is for background
input_shape = (300, 300, 3)

#######################################################################
#   Create model
#######################################################################
model = SSD(input_shape, num_classes=NUM_CLASSES)
# model.load_weights(
#     './weights/VGG16SSD300weights_voc_2007_class20.hdf5', by_name=True)
new_model = SSD(input_shape, num_classes=NUM_CLASSES)
model.load_weights(
    './weights/VGG16.h5', by_name=True)

# l_names = get_weights_layers("./weights/MobileNetSSD300weights_voc_2007_class20.hdf5")
# print(l_names)
# isWeightsChanged(model, new_model)

# model.summary()
# for L in model.layers:
#     print(str(L.name))

# %%
#######################################################################
#   Freeze model layers # TODO: figure out the layers to freeze
#######################################################################
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3',
          'conv4_1', 'conv4_2', 'conv4_3', 'pool4',
          'conv5_1', 'conv5_2', 'conv5_3', 'pool5']

for L in model.layers:
    if L.name in freeze:
        # print(L.name)
        L.trainable = False

# TODO: load entire model, rather than just weights. Might require custom objects

# %%
#######################################################################
#   Setup data generators for training
# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd7_training.ipynb
# https://github.com/pierluigiferrari/data_generator_object_detection_2d
#######################################################################
priors = pickle.load(open('priorFiles/prior_boxes_ssd300VGG16.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

# %%
# key = image_name (no path), value = [xmin, ymin, xmax, ymax]
gt = get_annotations("./dataset/annotations_raw.txt")
# TODO: sort this manually
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[: num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)


# %%
path_prefix = './dataset/images/'
batch_size = 8
gen = Generator(gt, bbox_util, batch_size, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)


# %%
#######################################################################
#   Setup callbacks and monitoring
#######################################################################
def schedule(epoch, decay=0.9):
    if epoch < 12:
        return base_lr
    elif epoch < 20:
        return base_lr
    elif epoch < 25:
        return base_lr * decay**(epoch/2)
    else:
        return base_lr * decay**(epoch)  # 0.00001
    # return base_lr * decay**(epoch) # original


model_checkpoint = ModelCheckpoint(
    './output/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True, period=5)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, verbose=1)  # TODO: fix this

# TODO: Fix Tensorboard variables
tensorboard = TensorBoard(log_dir='./logs/tensorboard/005', write_images=True)
lrSchedular = LearningRateScheduler(schedule)

# %%
#######################################################################
#  Instantiate optimizer, SDD loss function and Compile model
#######################################################################
# base_lr = 3e-4
# NOTE: 1e-2 is shit
base_lr = 4e-3
optim = Adam(lr=base_lr)
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # example for SDD7
# optim = RMSprop(lr=base_lr)
# optim = SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['acc'])

# %%
#######################################################################
#   TRAIN model
#######################################################################

# Train with frozen layers first, to get a stable loss.
nb_epoch = 12
callbacks = [tensorboard, model_checkpoint, lrSchedular]
# callbacks = [tensorboard, model_checkpoint, lrSchedular, early_stopping]
history = model.fit_generator(gen.generate(True), gen.train_batches,
                              verbose=1,
                              epochs=nb_epoch,
                              initial_epoch=0,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              validation_steps=gen.val_batches)
#       nb_val_samples=gen.val_batches,
#       nb_worker=1)
model.save_weights('./output/checkpoints/trained_weights_stage1_005.hdf5')
# model.save("./output/trained_model_final_005.h5")

# Unfreeze and continue training, to fine-tune.
for i in range(len(model.layers)):
    model.layers[i].trainable = True

base_lr = 3e-4
optim = Adam(lr=base_lr)
callbacks = [tensorboard, model_checkpoint, lrSchedular, early_stopping]

history = model.fit_generator(gen.generate(True), gen.train_batches,
                              verbose=1,
                              epochs=nb_epoch+50,
                              initial_epoch=nb_epoch,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              validation_steps=gen.val_batches)
#       nb_val_samples=gen.val_batches,
#       nb_worker=1)
model.save_weights('./output/trained_weights_final_005.hdf5')
model.save("./output/trained_model_final_005.h5")

#######################################################################
#   TODO: Confirm if this is valiation
#######################################################################
# %%
# inputs = []
# images = []
# img_path = path_prefix + sorted(val_keys)[0]
# img = image.load_img(img_path, target_size=(300, 300))
# img = image.img_to_array(img)
# images.append(imread(img_path))
# inputs.append(img.copy())
# inputs = preprocess_input(np.array(inputs))


# # %%
# preds = model.predict(inputs, batch_size=1, verbose=1)
# results = bbox_util.detection_out(preds)


# # %%
# for i, img in enumerate(images):
#     # Parse the outputs.
#     det_label = results[i][:, 0]
#     det_conf = results[i][:, 1]
#     det_xmin = results[i][:, 2]
#     det_ymin = results[i][:, 3]
#     det_xmax = results[i][:, 4]
#     det_ymax = results[i][:, 5]

#     # Get detections with confidence higher than 0.6.
#     top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

#     top_conf = det_conf[top_indices]
#     top_label_indices = det_label[top_indices].tolist()
#     top_xmin = det_xmin[top_indices]
#     top_ymin = det_ymin[top_indices]
#     top_xmax = det_xmax[top_indices]
#     top_ymax = det_ymax[top_indices]

#     colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

#     plt.imshow(img / 255.)
#     currentAxis = plt.gca()

#     for i in range(top_conf.shape[0]):
#         xmin = int(round(top_xmin[i] * img.shape[1]))
#         ymin = int(round(top_ymin[i] * img.shape[0]))
#         xmax = int(round(top_xmax[i] * img.shape[1]))
#         ymax = int(round(top_ymax[i] * img.shape[0]))
#         score = top_conf[i]
#         label = int(top_label_indices[i])
# #         label_name = voc_classes[label - 1]
#         display_txt = '{:0.2f}, {}'.format(score, label)
#         coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
#         color = colors[label]
#         currentAxis.add_patch(plt.Rectangle(
#             *coords, fill=False, edgecolor=color, linewidth=2))
#         currentAxis.text(xmin, ymin, display_txt, bbox={
#                          'facecolor': color, 'alpha': 0.5})

#     plt.show()
