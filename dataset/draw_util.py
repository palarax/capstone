import cv2
import numpy as np


def drawBbox(image, points):
    """Draw bounding box in image

    Parameters:
        image (cv2Img): image to draw on
        points (list): list of points (x and y coordinates), e.g. [[x1, y1], [x2, y2]]

    Returns:
        cv2Img: updated image with bbox
    """
    # draw green rectange with thickness of 3
    cv2.rectangle(image, tuple(points[0]), tuple(points[1]), (0, 255, 0), 3)
    return image


def drawPolygon(image, points):
    """Draw Polygon in image

    Parameters:
        image (cv2Img): image to draw on
        points (list): list of poinst (x and y coordinates), 
                e.g. [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]

    Returns:
        cv2Img: updated image with polygon
    """

    # reformat points
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], True, (0, 255, 0), 3)
    return image



if __name__ == "__main__":
    import json
    import argparse

    parser = argparse.ArgumentParser(
        description='Display images annotation mask')
    parser.add_argument(
        '-i', '--image', help='image to select', required=True)
    parser.add_argument('-a', '--annotation',
                        help='image annotation', required=True)
    args = parser.parse_args()

    # using supervisely annotation
    annot = json.load(open(args.annotation))

    image = cv2.imread(args.image, 1)    # read image in original color

    for obj in annot["objects"]:
        points = obj["points"]["exterior"]  # list of points
        #image = drawBbox(image, points)
        drawPolygon(image, points)

    cv2.imshow("Image", image)
    cv2.waitKey(0)  # 0==wait forever
