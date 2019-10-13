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


def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

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

    # Low, Medium, High, Extreme
    classes = ['low', 'medium', 'high']
    
    leftLine = [(int(width*portion), height), (int(width*portion), 0)]
    rightLine = [(int(width*(1-portion)), height), (int(width*(1-portion)), 0)]

    # detect_faces_haar(frame)
    cv2.line(frame,leftLine[0],leftLine[1], (255,0,0),5)
    cv2.line(frame,rightLine[0],rightLine[1], (255,0,0),5)

    for obj in prediction[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        # [0]class/risk  [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax  [6]distance [7] ratio in screen

        # RAW: int(round(obj[2] * width / img_width))
        xmin = int(round(obj[2]))
        ymin = int(round(obj[3]))
        xmax = int(round(obj[4]))
        ymax = int(round(obj[5]))

        inz = isInZone(obj, int(width*portion), (int(width*(1-portion)) ) )
        risk = analyse_risk(obj[0], obj[6], obj[7], inz)
        # if inz:
        #     risk = 'high'
        # else:
        #     risk = 'medium'

        icon = ICONS[risk]
        color = db.get_signal(risk)
        labels = ['background','person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
        risk = labels[int(obj[0])]# TODO Remove

        # boxes.append([xmin, ymin, xmax, ymax])  # record obj box

        logging.debug("Class[%s] Conf[%.2f] xmin[%d] ymin[%d] xmax[%d] ymax[%d]",
                      risk, obj[1], xmin, ymin, xmax, ymax)
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


def analyse_risk(id_class, distance, ratio, in_zone,speed=50):

    # Low, Medium, High, Extreme
    classes = ['low', 'medium', 'high']
    # [0]class   [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax  [6]distance [7] ratio in screen
    risk = 'medium'
    stp_dist = 300
    if in_zone:
        return 'high'
        
    if stp_dist > distance:
        return 'high'
    
    if float(ratio) > 0.20:
        return 'high'

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
    # ct = CentroidTracker()
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
