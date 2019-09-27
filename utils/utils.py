import base64
import io
import cv2
from imageio import imread
import logging


def distance_to_object(known_height, object_height, focal_length):
    '''
    Using "triangle similarity" or ratio to calculate the
    distance to an object\n
    :param int known_height: known height of the object in real life (mm)\n
    :param int object_height: height of perceived object in pixels\n
    :return: distance to object in cm
    '''
    # focal length = (Height Pixels * distance to object) / actual height
    distance_cm = (known_height * focal_length) / object_height
    return distance_cm  # m
    # raise NotImplementedError("Stub")


def configure_icons(db, icons, dimensions):
    '''
    Decodes base64 strings to icon images, and sets them to dimensions
    :param Xaidb db: database instance
    :param dimensions: width and height values (square)
    :return: dictionary of Icons
    '''
    all_icons = db.get_all_icons()
    # ICONS
    for icon in all_icons:
        logging.debug("[utils] configuring icon [%s]", icon)
        if len(all_icons[icon]) < 2:
            logging.debug("[utils] No data for icon [%s]", icon)
            continue
        decoded_str = base64.b64decode(all_icons[icon])
        icon_data = imread(io.BytesIO(decoded_str))
        cv2_img = cv2.cvtColor(icon_data, cv2.COLOR_BGR2RGBA)
        icons[icon] = cv2.resize(cv2_img, (dimensions, dimensions))
