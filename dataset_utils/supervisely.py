import numpy as np
import os
import pickle
import json

LABELS = ["pedestrian_bb", "cyclist_bb",
          "facing_back_bb", "on-call_bb", "on-mobile_bb"]


def parse_supervisely_annotation(root="dataset/bbox_transform", ann_dir, img_dir, cache_name, labels=LABELS):
    """
    Parse supervisely directory
    """
    all_insts = []
    seen_labels = {}
    for d in ["cyclist", "pedestrians_on_phone", "person_backwords", "people_mobile"]:
        annotations_dir = os.path.join(root, d, "ann")
        img_dir = os.path.join(root, d, "img/")
        _parse_to_yolo(annotations_dir, img_dir,all_insts, seen_labels)

    cache = {'all_insts': all_insts, 'seen_labels': seen_labels}
    with open(cache_name, 'wb') as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)    
                        
    return all_insts, seen_labels



def _parse_to_yolo(ann_dir, img_dir, all_insts, seen_labels, labels=LABELS):
    """
    Parse supervisely annotations to yolo
    """
    if False:
        with open(cache_name, 'rb') as handle:
            cache = pickle.load(handle)
        all_insts, seen_labels = cache['all_insts'], cache['seen_labels']
    else:
        
        for ann in sorted(os.listdir(ann_dir)):
            annotation = os.path.join(ann_dir,ann)
            img = {'object':[]}
            annot = json.load(open(annotation))
            img['filename'] = img_dir + ann[:-5]
            img['width'] = annot["size"]["width"]
            img['height'] = annot["size"]["height"]
            for obje in annot["objects"]:
                obj = {}
                points = obje["points"]["exterior"]
                obj['name'] = obje["classTitle"]
                if obj['name'] in seen_labels:
                    seen_labels[obj['name']] += 1
                else:
                    seen_labels[obj['name']] = 1
                obj['xmin'] = int(round(float(points[0][0])))
                obj['ymin'] = int(round(float(points[0][1])))
                obj['xmax'] = int(round(float(points[1][0])))
                obj['ymax'] = int(round(float(points[1][1])))
                img['object'] += [obj]
            
            all_insts += [img]   
                        
    # return all_insts, seen_labels




if __name__ == "__main__":
    root = "dataset/bbox_transform"
    for d in ["cyclist", "pedestrians_on_phone", "person_backwords"]:
        annotations_dir = os.path.join(root, d, "ann")
        img_dir = os.path.join(root, d, "img/")
        parse_annotation(annotations_dir, img_dir,"cache.cache")

