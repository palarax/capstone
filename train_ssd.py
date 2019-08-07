
#%%
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

# Deep learning
import keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# from keras import backend as K
from keras.models import Model
from keras.preprocessing import image

# Custom classes and libraries
from model.ssd300MobileNet import SSD
from utils.ssd_training import MultiboxLoss
from utils.ssd_utils import BBoxUtility
from utils.generator import Generator

########## VALIDATION
from scipy.misc import imread
from keras.applications.imagenet_utils import preprocess_input

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (8, 8)
# plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True)
#######################################################################
#   Setup GPU
#######################################################################
SESSION = tf.Session(config=tf.ConfigProto(log_device_placement=True))
set_session(SESSION)
# K.clear_session() # Clear previous models from memory.
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# set_session(tf.Session(config=config))

#%%
# some constants
NUM_CLASSES = 4 # positive clases
input_shape = (300, 300, 3)

#%%
#######################################################################
#   Create model
#######################################################################
model = SSD(input_shape, num_classes=NUM_CLASSES)
model.load_weights('VGG16SSD300_weights_voc_2007.hdf5', by_name=True)

#%%
#######################################################################
#   Freeze model layers
#######################################################################
freeze = ['input_1', 'conv1_1', 'conv1_2', 'pool1',
          'conv2_1', 'conv2_2', 'pool2',
          'conv3_1', 'conv3_2', 'conv3_3', 'pool3']#,
#           'conv4_1', 'conv4_2', 'conv4_3', 'pool4']

for L in model.layers:
    if L.name in freeze:
        L.trainable = False

# TODO: load entire model, rather than just weights. Might require custom objects
#%%
#######################################################################
#  Instantiate optimizer, SDD loss function and Compile model
#######################################################################
base_lr = 3e-4
optim = keras.optimizers.Adam(lr=base_lr)
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # example for SDD7
# optim = keras.optimizers.RMSprop(lr=base_lr)
# optim = keras.optimizers.SGD(lr=base_lr, momentum=0.9, decay=decay, nesterov=True)
model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['acc'])

#%%
#######################################################################
#   Setup data generators for training
# https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd7_training.ipynb
# https://github.com/pierluigiferrari/data_generator_object_detection_2d
#######################################################################
priors = pickle.load(open('priorFiles/prior_boxes_ssd300VGG16.pkl', 'rb'))
bbox_util = BBoxUtility(NUM_CLASSES, priors)

#%%
gt = pickle.load(open('voc_2007.pkl', 'rb'))
keys = sorted(gt.keys())
num_train = int(round(0.8 * len(keys)))
train_keys = keys[:num_train]
val_keys = keys[num_train:]
num_val = len(val_keys)


#%%
path_prefix = 'path2yourJPEG'
gen = Generator(gt, bbox_util, 16, path_prefix,
                train_keys, val_keys,
                (input_shape[0], input_shape[1]), do_crop=False)




#%%
#######################################################################
#   Setup callbacks and monitoring
#######################################################################
def schedule(epoch, decay=0.9):
    return base_lr * decay**(epoch)

model_checkpoint = keras.callbacks.ModelCheckpoint('./output/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True, period=1)
# early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10, verbose=1) #TODO: fix this
#TODO: tensorboard
lrSchedular =  keras.callbacks.LearningRateScheduler(schedule)
callbacks = [model_checkpoint, lrSchedular]


#%%
#######################################################################
#   TRAIN model
#######################################################################
nb_epoch = 30
history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                              callbacks=callbacks,
                              validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)


#######################################################################
#   TODO: Confirm if this is valiation
#######################################################################
#%%
inputs = []
images = []
img_path = path_prefix + sorted(val_keys)[0]
img = image.load_img(img_path, target_size=(300, 300))
img = image.img_to_array(img)
images.append(imread(img_path))
inputs.append(img.copy())
inputs = preprocess_input(np.array(inputs))


#%%
preds = model.predict(inputs, batch_size=1, verbose=1)
results = bbox_util.detection_out(preds)


#%%
for i, img in enumerate(images):
    # Parse the outputs.
    det_label = results[i][:, 0]
    det_conf = results[i][:, 1]
    det_xmin = results[i][:, 2]
    det_ymin = results[i][:, 3]
    det_xmax = results[i][:, 4]
    det_ymax = results[i][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]

    colors = plt.cm.hsv(np.linspace(0, 1, 4)).tolist()

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for i in range(top_conf.shape[0]):
        xmin = int(round(top_xmin[i] * img.shape[1]))
        ymin = int(round(top_ymin[i] * img.shape[0]))
        xmax = int(round(top_xmax[i] * img.shape[1]))
        ymax = int(round(top_ymax[i] * img.shape[0]))
        score = top_conf[i]
        label = int(top_label_indices[i])
#         label_name = voc_classes[label - 1]
        display_txt = '{:0.2f}, {}'.format(score, label)
        coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
        color = colors[label]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
    
    plt.show()


