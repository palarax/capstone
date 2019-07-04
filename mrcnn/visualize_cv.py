import cv2
import numpy as np


def random_colors(n):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    np.random.seed(1)  # initialize random number
    colors = [tuple(255*np.random.rand(3)) for _ in range(n)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    ids:
    names:
    scores: confidence scores for each box
    """
    n_instances = boxes.shape[0]  # objects in the fame

    if not n_instances:
        print("No instances to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    colors = random_colors(n_instances)
    # height, width = image.shape[:2] # what is this

    for i, color in enumerate(colors):
        if not np.any(boxes[i]):
            continue  # continue if no boxes found
        # apply boxes, mask and caption/label to image

        y_1, x_1, y_2, x_2 = boxes[i]  # box coordinates
        mask = masks[:, :, i]
        image = apply_mask(image, mask, color)  # image with mask on it

        image = cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color, 2)
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        image = cv2.putText(
            image, caption, (x_1, y_1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image

## For testing
if __name__ == '__main__':
    import os
    import coco
    import model as modellib

    ROOT_DIR = os.getcwd()
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 2

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    capture = cv2.VideoCapture("road.mp4") # using web cam
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

    # only activate whilst the video is available
    while True:
        return_value, frame = capture.read()

        results = model.detect([frame], verbose=0) # boxes, labels, etc

        r = results[0] # first thing in results ??
 
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )

        cv2.imshow('frame', frame) # display frame

        # Kill window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    capture.release()
    cv2.destroyAllWindows()
