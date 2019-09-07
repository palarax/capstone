# import os
from timeit import default_timer as timer
import cv2
import numpy as np

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

from XAI.keras_loss_function.keras_ssd_loss import SSDLoss
from XAI.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from XAI.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from XAI.keras_layers.keras_layer_L2Normalization import L2Normalization
from XAI.ssd_encoder_decoder.ssd_output_decoder import decode_detections_fast


def load_ssd_model():

    model_path = 'SSD_MODEL.h5'
    # We need to create an SSDLoss object in order to pass that to the model loader.
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
    K.clear_session()  # Clear previous models from memory.
    model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
    return model


def get_fps(accum_time, curr_fps, prev_time, fps):
    curr_time = timer()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        fps = "FPS: " + str(curr_fps)
        curr_fps = 0
    return accum_time, curr_fps, prev_time, fps


def detect_image(model, frame, vidw, vidh):
    img_height = 300
    img_width = 300
    class_colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255)]
    classes = ['background', 'cyclist', 'risk']

    img = cv2.resize(frame, (img_height, img_width))
    inp_img = [image.img_to_array(img)]
    tmp_inp = np.array(inp_img)
    preprocessed_inp = preprocess_input(tmp_inp)
    y_pred = model.predict(preprocessed_inp)

    y_pred_decoded = decode_detections_fast(y_pred,
                                            confidence_thresh=0.5,
                                            iou_threshold=0.4,
                                            top_k=200,
                                            normalize_coords=True,
                                            img_height=img_height,
                                            img_width=img_width)

    height, width, _ = frame.shape

    for box in y_pred_decoded[0]:
        # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
        # [0]class   [1]conf  [2]xmin   [3]ymin   [4]xmax   [5]ymax
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        conf = int(box[1])

        xmin = int(round(box[2] * width / img_width))
        ymin = int(round(box[3] * height / img_height))
        xmax = int(round(box[4] * width / img_width))
        ymax = int(round(box[5] * height / img_height))

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                      class_colors[int(box[0])], 3)

        text_top = (xmin, ymin-10)
        text_bot = (xmin + 80, ymin + 5)
        text_pos = (xmin + 5, ymin)
        cv2.rectangle(frame, text_top, text_bot, class_colors[int(box[0])], -1)
        cv2.putText(frame, label, text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    return frame


def detect_video(model, video_path, output_path="output"):
    vid = cv2.VideoCapture(video_path)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    vid.set(cv2.CAP_PROP_FPS, 30)
    vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        return_value, frame = vid.read()

        if not return_value:
            break

        image = detect_image(model, frame, vidw, vidh)

        accum_time, curr_fps, prev_time, fps = get_fps(accum_time, curr_fps, prev_time, fps)
        # Draw FPS in top left corner
        cv2.rectangle(image, (0, 0), (50, 17), (255, 255, 255), -1)
        cv2.putText(image, fps, (3, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.imshow("SSD results", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = load_ssd_model()
    detect_video(model, "dataset/test.mp4")
