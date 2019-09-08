
def distance_to_object(known_height, object_height, focal_length=12):
    '''
    Using "triangle similarity" or ratio to calculate the
    distance to an object\n
    :param int known_height: known height of the object in real life (mm)\n
    :param int object_height: height of perceived object in pixels\n
    :return: distance to object in mm
    '''
    # focal length = (Height Pixels * distance to object) / actual height
    known_height = 165 * 100  # average person height (mm)
    return (known_height * focal_length) / object_height
    # raise NotImplementedError("Stub")
