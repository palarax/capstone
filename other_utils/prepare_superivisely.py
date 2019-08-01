import os
import json

LABELS = ["pedestrian_bb", "cyclist_bb", "on-call_bb", "on-mobile_bb"]


def generate_annotation(annotation, img, aFile):
    """
    Generate annotation file from json
    <object-class-id> <center-x> <center-y> <width> <height>
    """
    ann_list = []
    annot = json.load(open(annotation))
    width, height = annot["size"]["width"], annot["size"]["height"]

    for obj in annot["objects"]:
        # [[left, top], [right, bottom]]
        # [[x1, y1], [x2, y2]]
        if obj["classTitle"] not in LABELS:
            continue
        classId = LABELS.index(obj["classTitle"])
        points = obj["points"]["exterior"]  # list of points
        # bbox_w = points[1][0] - points[0][1]
        # bbox_h = points[1][1] - points[0][1]
        # x = (points[0][0] + (bbox_w / 2)) / width
        # x = (points[0][1] + (bbox_h / 2)) / height
        # bbox_w = bbox_w / width
        # bbox_h = bbox_h / height
        info = "%s,%d,%d,%d,%d,%d" % (
            img, points[0][0], points[0][1], points[1][0], points[1][1], classId)
        print(info)
        aFile.write(info + '\n')


if __name__ == "__main__":
    root = "bbox_transform"
    new_img_dir = "dataset/images"
    # new_annotations = os.path.join("../dataset", "annotations.txt")
    
    counter = 0
    aFile = open(os.path.join("../dataset", "annotations.txt"),'w')
    for d in ["cyclist", "pedestrians_on_phone", "person_backwords","people_mobile"]:
        annotations_dir = os.path.join(root, d, "ann")
        img_dir = os.path.join(root, d, "img")

        for ann in os.listdir(annotations_dir):
            img = os.path.join(img_dir, ann[:-5])
            new_img_name = "img_{}.jpg".format(counter)
            # move image
            os.rename(img, os.path.join("../dataset/images", new_img_name))
            annotation = os.path.join(annotations_dir, ann)
            
            generate_annotation(annotation, os.path.join("dataset/images", new_img_name), aFile)
            counter += 1

    aFile.close()