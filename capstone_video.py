# import os
from timeit import default_timer as timer
import os
import sys
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

from utils.utils import configure_icons, transparentOverlay
from db_utils.Xaidb import Xaidb

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


def isInZone(obj, z1, z2):
    w1 = obj[2]
    w2 = obj[4]

    if (w1 < z1 < w2) or (w1 < z2 < w2) or (w1 > z1 and w2 < z2):
        return True

    return False


def draw_objects(prediction, frame, classes, portion=0.3):
    '''Draw objects in image frame
    '''
    class_colors = [[0, 0, 0], [0, 128, 255], [0, 0, 255]]
    height, width, _ = frame.shape
    boxes = []

    leftLine = [(int(width*portion), height), (int(width*portion), 0)]
    rightLine = [(int(width*(1-portion)), height), (int(width*(1-portion)), 0)]

    # detect_faces_haar(frame)
    cv2.line(frame, leftLine[0], leftLine[1], (255, 0, 0), 5)
    cv2.line(frame, rightLine[0], rightLine[1], (255, 0, 0), 5)

    for obj in prediction[0]:
        # if int(obj[0]) not in [1, 2,4, 18,19,20]:
        #     continue
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        # [0]class/risk  [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax  [6]distance [7] ratio in screen

        # RAW: int(round(obj[2] * width / img_width))
        xmin = int(round(obj[2]))
        ymin = int(round(obj[3]))
        xmax = int(round(obj[4]))
        ymax = int(round(obj[5]))

        inz = isInZone(obj, int(width*portion), (int(width*(1-portion))))
        risk, danger_level = analyse_risk(obj[0], obj[6], obj[7], inz)

        if danger_level == 1:
            # icon = ICONS["low"]
            icon = "-1"
            color = db.get_signal("low")
        elif danger_level == 2:
            icon = ICONS["medium"]
            color = db.get_signal("medium")
        else:
            icon = ICONS["high"]
            color = db.get_signal("high")

        logging.debug("Class[%s] Conf[%.2f] Risk [%s]",
                      obj[0], obj[1], risk)
        logging.debug("Distance [%f] Portion[%f]", obj[6], obj[7])

        conf = float("{:.2f}".format(obj[1]))
        label = '{}: {}'.format(
            risk, conf)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      color, 3)

        text_top = (xmin, ymin+20)
        # text_bot = (xmin + 150, ymin - 10)
        text_bot = (xmin + (xmax-xmin), ymin)
        text_pos = (xmin, ymin+10)
        cv2.rectangle(frame, text_top, text_bot, color, -1)
        cv2.putText(frame, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2)

        if danger_level == 1:
            # if True:
            continue

        x_offset = xmax - 50
        y_offset = ymin - 60

        if icon != "-1":
            frame = transparentOverlay(frame, icon, pos=(x_offset, y_offset))

    return boxes


def detect_faces_haar(frame):
    face_cascade = cv2.CascadeClassifier(
        'configuration/haarcascade_frontalface_default.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    # faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10,
                                          minSize=(75, 75))
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # return frame


def analyse_risk(id_class, distance, ratio, in_zone):

    # Low, Medium, High, Extreme
    classes = ['Low Risk', 'High Risk', 'Danger']
    # [0]class   [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax  [6]distance [7] ratio in screen
    risk = 'Low Risk'
    danger_level = 1

    if float(ratio) > 0.12:
        danger_level = 2
        risk = "High risk of collision"

    if distance > 400 or float(ratio) > 0.5:
        danger_level = 3
        risk = "DANGER: Too close"

    if in_zone:
        danger_level = 4
        risk = "DANGER: PERSON IN FRONT"

    return risk, danger_level


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
    global bytes

    vid = cv2.VideoCapture(video_path)
    logging.info("Starting Video Stream")
    if not vid.isOpened():
        video_path = "10.0.0.28:8080"

        logging.debug("Couldn't open local webcam or video")
        stream = 'http://' + video_path + '/video'
        logging.debug('Streaming from: %s', stream)
        # Open ByteStram
        stream = urllib2.urlopen(stream)
        bytes = bytes()

    vid.set(1, 100)  # Skip first few frames cause they are garbage
    while True:
        frame = None
        if not vid.isOpened():
            bytes += stream.read(1024)
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                frame = cv2.imdecode(np.fromstring(
                    jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                continue
        else:
            return_value, frame = vid.read()

            if not return_value:
                break

        # frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) # resize both axis by half
        frame = cv2.resize(frame, (1200, 900))

        # if int(vid.get(cv2.CAP_PROP_POS_FRAMES)) % skip == 0:  # every 3 frames
        if True:
            predictions = get_prediction(
                model, frame, confidence_thresh, iou_threshold, img_height, img_width)
            # TODO: implement tracking

            draw_objects(predictions, frame, class_labels)

        # Calculate and draw FPS
        accum_time, curr_fps, prev_time, fps = calculate_fps(
            accum_time, curr_fps, prev_time, fps, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow("SSD results", frame)

    # vid.release()
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

    # process_video(model, config["model_processing"])
    process_video(model, config["model_processing"], config["video_path"])


if __name__ == "__main__":
    main()
