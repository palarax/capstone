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

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    """
    :param src: Input Color Background Image
    :param overlay: transparent Image (BGRA)
    :param pos:  position where the image to be blit.
    :param scale : scale factor of transparent image.
    :return: Resultant Image
    """
    h, w, _ = overlay.shape  # Size of foreground

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)

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