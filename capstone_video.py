# import os
from timeit import default_timer as timer
import os
import json
import urllib.request as urllib2  # VoIP camera
import logging
import logging.config
import numpy as np
import cv2
import tensorflow as tf

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from XAI.keras_loss_function.keras_ssd_loss import SSDLoss
from XAI.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from XAI.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from XAI.keras_layers.keras_layer_L2Normalization import L2Normalization
from XAI.ssd_encoder_decoder.ssd_output_decoder import decode_detections_fast

from utils.utils import configure_icons
from db_utils.Xaidb import Xaidb
from centroidtracker.centroidtracker import CentroidTracker

# TF debug
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.Session(config=tf.ConfigProto(log_device_placement=True))

# GLOBALS
db = None
ICONS = {}


def setup_log(log_config):
    '''Setup logging
    '''
    with open(log_config, encoding='utf-8-sig') as conf_file:
        jc = json.load(conf_file)
        logging.config.dictConfig(jc["main_app"])


def calculate_fps(accum_time, curr_fps, prev_time, fps, image):
    '''Calculate current FPS
    '''
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0
    # Draw FPS in top left corner
    cv2.rectangle(image, (0, 0), (50, 17), (255, 255, 255), -1)
    cv2.putText(image, fps, (3, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    return accum_time, curr_fps, prev_time, fps


def get_color(classes):
    # TODO: implement through DB
    # ["background", "cyclist", "risk"],
    # BGR colors
    BACKGROUND = (0, 0, 0)
    DANGER = (0, 0, 255)  # RED
    WARNING = (0, 128, 255)  # ORANGE
    CAUTION = (0, 255, 255)  # YELLOW
    NO_IMMEDIATE_DANGER = (255, 0, 0)  # BLUE
    return [(0, 0, 0), DANGER, WARNING]


def load_ssd_model(model_path="SSD_MODEL.h5"):
    '''Load Pretrained SSD model
    '''
    logging.info("Loading SSD model [%s]", model_path)
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    K.clear_session()  # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
    return model


def get_prediction(model, frame, confidence_thresh, iou_threshold, img_height=300, img_width=300):
    '''Get model prediction
    return decoded prediction
    '''
    height, width, _ = frame.shape
    img = cv2.resize(frame, (img_height, img_width))
    inp_img = [image.img_to_array(img)]
    tmp_inp = np.array(inp_img)
    preprocessed_inp = preprocess_input(tmp_inp)
    prediction = model.predict(preprocessed_inp)
    return decode_detections_fast(prediction,
                                  confidence_thresh=confidence_thresh,
                                  iou_threshold=iou_threshold,
                                  top_k=200,
                                  normalize_coords=True,
                                  img_height=height,
                                  img_width=width)


def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]    # Position of foreground/overlay image

    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x+i >= rows or y+j >= cols:
                continue
            alpha = float(overlay[i][j][3]/255.0)  # read the alpha channel
            src[x+i][y+j] = alpha*overlay[i][j][:3]+(1-alpha)*src[x+i][y+j]
    return src


def draw_objects(prediction, frame, classes):
    '''Draw objects in image frame
    '''
    class_colors = [[0, 0, 0], [0, 128, 255], [0, 0, 255]]
    height, width, _ = frame.shape
    boxes = []

     # Low, Medium, High, Extreme
    classes = ['low', 'medium', 'high']
    # Blending the images with 0.3 and 0.7

    for obj in prediction[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        # [0]class/risk  [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax  [6]distance [7] ratio in screen

        # RAW: int(round(obj[2] * width / img_width))
        xmin = int(round(obj[2]))
        ymin = int(round(obj[3]))
        xmax = int(round(obj[4]))
        ymax = int(round(obj[5]))
        
        risk = analyse_risk(obj[0], obj[6], obj[7])

        icon = ICONS[risk]
        color = db.get_signal(risk)

        boxes.append([xmin, ymin, xmax, ymax])  # record obj box

        logging.debug("Class[%s] Conf[%.2f] xmin[%d] ymin[%d] xmax[%d] ymax[%d]", risk, obj[1], xmin, ymin, xmax, ymax)
        logging.debug("Distance [%f] Portion[%f]", obj[6], obj[7])

        conf = float("{:.2f}".format(obj[1]))
        label = '{}: {:.2f}'.format(
            risk, obj[1])

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      color, 3)

        text_top = (xmin, ymin-10)
        text_bot = (xmin + 80, ymin + 5)
        text_pos = (xmin + 5, ymin)
        cv2.rectangle(frame, text_top, text_bot, color, -1)
        cv2.putText(frame, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1)


        if risk == 'low':
            continue

        x_offset = xmax - 50
        y_offset = ymin - 50

        frame = transparentOverlay(frame, icon, (x_offset, y_offset))

    return boxes


def take_action():
    raise NotImplementedError("Stub")


def analyse_risk(id_class, distance, ratio):

    # Low, Medium, High, Extreme
    classes = ['low', 'medium', 'high']
    # [0]class   [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax  [6]distance [7] ratio in screen
    risk = 'medium'
    if float(ratio) > 0.20:
        risk = 'high'
    
    return risk


def process_video(model, config, video_path=0, skip=1):
    '''Process video or camera stream
    video_path: file or (0 or -1) for video stream
    '''

    confidence_thresh = config["confidence_thresh"]
    iou_threshold = config["iou_threshold"]
    img_height = config["img_height"]
    img_width = config["img_width"]
    class_labels = config["labels"]

    # Setup values to calculate FPS
    accum_time, curr_fps = 0, 0
    fps = "FPS: ??"
    prev_time = timer()

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    logging.info("Starting Video Stream")

    # ==============================
    # Multi tracker init
    # trackers = cv2.MultiTracker_create()
    ct = CentroidTracker()
    # ==================================

    # vid.set(1, 100)  # Skip first few frames cause they are garbage
    while True:
        return_value, frame = vid.read()

        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # resize both axis by half
        if not return_value:
            break

        frame = cv2.resize(frame, (1200, 900))

        if int(vid.get(cv2.CAP_PROP_POS_FRAMES)) % skip == 0:  # every 3 frames
            predictions = get_prediction(
                model, frame, confidence_thresh, iou_threshold, img_height, img_width)
            # TODO: implement tracking

            boxes = draw_objects(predictions, frame, class_labels)
            # get centroid point of box
            # cX = int((startX + endX) / 2.0)
            # cY = int((startY + endY) / 2.0)
            # objects = ct.update(boxes)
            # loop over the tracked objects
            # for (objectID, centroid) in objects.items():
            #     # draw both the ID of the object and the centroid of the
            #     # object on the output frame
            #     text = "ID {}".format(objectID)
            #     cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #     logging.debug(centroid)
            #     cv2.circle(
            #         frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # Calculate and draw FPS
        accum_time, curr_fps, prev_time, fps = calculate_fps(
            accum_time, curr_fps, prev_time, fps, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow("SSD results", frame)

    vid.release()
    cv2.destroyAllWindows()


def main(log_config="configuration/log_config.json", main_config="configuration/config.json"):
    """ Main Function
    """
    global db
    setup_log(log_config)
    config = {}
    with open(main_config, encoding='utf-8-sig') as conf_file:
        config = json.load(conf_file)
    model = load_ssd_model(config["model_processing"]["file"])

    # init database and icons
    db = Xaidb(config["database"]["name"])
    configure_icons(db, ICONS, config["icons_dimensions"])

    process_video(model, config["model_processing"])
    # process_video(model, config["model_processing"], config["video_path"])


if __name__ == "__main__":
    main()
