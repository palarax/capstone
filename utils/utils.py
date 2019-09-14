
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
    return distance_cm # m
    # raise NotImplementedError("Stub")
