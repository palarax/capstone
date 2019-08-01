import os
import json

def generate_yolo_annotation(annotation, img, labels, new_annot):
    """
    Generate annotation file from json
    yolo annotations
    img_path x_min,y_min,x_max,y_max,class_id
    """
    annot = json.load(open(annotation))
    for obj in annot["objects"]:
        # [[left, top], [right, bottom]]
        # [[x1, y1], [x2, y2]]
        class_id = labels.index(obj["classTitle"])
        points = obj["points"]["exterior"]  # list of points
        info = "%s %d,%d,%d,%d,%d" % (
            img, points[0][0], points[0][1], points[1][0], points[1][1], class_id)
        new_annot.append(info)

if __name__ == "__main__":
    labels = ["pedestrian_bb", "cyclist_bb", "on-call_bb", "on-mobile_bb", "facing_back_bb"]
    root = "bbox_transform"
    new_img_dir = "dataset/images"
    counter = 0
    ann_list = []
    for d in ["cyclist", "pedestrians_on_phone", "person_backwords", "people_mobile"]:
        annotations_dir = os.path.join(root, d, "ann")
        img_dir = os.path.join(root, d, "img")

        for ann in os.listdir(annotations_dir):
            img = os.path.join(img_dir, ann[:-5])
            new_img_name = "img_{}.jpg".format(counter)
            # move image
            os.rename(img, os.path.join("dataset/images", new_img_name))
            annotation = os.path.join(annotations_dir, ann)
            generate_yolo_annotation(annotation, os.path.join("dataset/images", new_img_name), labels, ann_list)
            counter += 1

    with open('dataset/annotations.txt', 'w') as f:
        for annot in ann_list:
            f.write(annot + '\n')