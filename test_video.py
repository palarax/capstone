# import os
from timeit import default_timer as timer
import os
import json
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

# TF debug
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.Session(config=tf.ConfigProto(log_device_placement=True))

def setup_log(log_config):
    '''Setup logging
    '''
    with open(log_config, encoding='utf-8-sig') as conf_file:
        jc = json.load(conf_file)
        logging.config.dictConfig(jc)

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
    WARNING = (0, 128, 255) # ORANGE
    CAUTION = (0, 255, 255) # YELLOW
    NO_IMMEDIATE_DANGER = (255, 0, 0) # BLUE
    return [(0, 0, 0), DANGER, WARNING]

def load_ssd_model(model_path="SSD_MODEL.h5"):
    '''Load Pretrained SSD model
    '''
    logging.info("Loading SSD model %s", model_path)
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
                                  img_height=img_height,
                                  img_width=img_width)

def draw_objects(prediction, frame, classes, img_height=300, img_width=300):
    '''Draw objects in image frame
    '''
    class_colors = [[0, 0, 0],[0, 128, 255],[0, 0, 255]]
    height, width, _ = frame.shape

    for obj in prediction[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        # [0]class   [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax
        label = '{}: {:.2f}'.format(classes[int(obj[0])], obj[1])
        conf = int(obj[1])

        xmin = int(round(obj[2] * width / img_width))
        ymin = int(round(obj[3] * height / img_height))
        xmax = int(round(obj[4] * width / img_width))
        ymax = int(round(obj[5] * height / img_height))

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      class_colors[int(obj[0])], 3)

        text_top = (xmin, ymin-10)
        text_bot = (xmin + 80, ymin + 5)
        text_pos = (xmin + 5, ymin)
        cv2.rectangle(frame, text_top, text_bot, class_colors[int(obj[0])], -1)
        cv2.putText(frame, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    return frame

def process_video(model, config, video_path=0):
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

    vid.set(cv2.CAP_PROP_FPS, 30)

    while True:
        return_value, frame = vid.read()

        if not return_value:
            break

        predictions = get_prediction(
            model, frame, confidence_thresh, iou_threshold, img_height, img_width)
        
        # TODO: implement IoU
        # TODO: implement risk analysis
        image = draw_objects(
            predictions, frame, class_labels, img_height, img_width)

        accum_time, curr_fps, prev_time, fps = calculate_fps(
            accum_time, curr_fps, prev_time, fps, image)

        cv2.imshow("SSD results", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def main(log_config="configuration/log_config.json", main_config="configuration/config.json"):
    """ Main Function
    """
    setup_log(log_config)
    config = {}
    with open(main_config, encoding='utf-8-sig') as conf_file:
        config = json.load(conf_file)
    model = load_ssd_model(config["model_processing"]["file"])

    process_video(model, config["model_processing"])
    # process_video(model, config["model_processing"], config["video_path"])


if __name__ == "__main__":
    main()
