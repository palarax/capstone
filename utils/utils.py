import numpy as np


def get_annotations(annot_raw):
    ''' Get annotations and output a dictionary of values
    #TODO: figure out what is the split of objects in images
    '''
    data = {}
    final_data = dict()
    for line in open(annot_raw):
        if "img_144" in line:
            continue # skip grayscale images
        # img, meta = line.split(" ")
        # img = img.replace("images/", "")  # remove path
        img, x1, y1, x2, y2, class_no = line.split(",")
        class_no = int(class_no) + 1  # account for background
        if class_no == 5:
            continue  # skip this class
        meta_data = [float(x1), float(y1), float(x2), float(y2), class_no]

        if img not in data:
            data[img] = [meta_data]
        else:
            data[img].append(meta_data)

    for key in data:
        bounding_boxes = []
        classes = []
        for box in data[key]:
            bounding_box = [box[0], box[1], box[2], box[3]]
            bounding_boxes.append(bounding_box)  # bounding boxes
            classes.append(_to_one_hot(box[4]))  # class
        bounding_boxes = np.asarray(bounding_boxes)
        classes = np.asarray(classes)
        image_data = np.hstack((bounding_boxes, classes))
        final_data[key] = image_data

    return final_data


def _to_one_hot(name):
    one_hot_vector = [0] * 4
    # background = 0
    if name == 1:
        one_hot_vector[0] = 1 # pedestrian
    elif name == 2:
        one_hot_vector[1] = 1 # cyclist
    elif name == 3:
        one_hot_vector[2] = 1 # on-call
    elif name == 4:
        one_hot_vector[3] = 1 # on-mobile
    else:
        print('unknown label: %s' % name)

    return one_hot_vector


def isWeightsChanged(model, model_new):
    print("Are weights equal - ",
          any([np.array_equal(a1, a2) for a1, a2 in zip(
              model.get_weights(), model_new.get_weights()[:4])]))


def get_weights_layers(weights):
    #Debug
    import h5py
    f = h5py.File(weights, 'r')
    return list(f.keys())
