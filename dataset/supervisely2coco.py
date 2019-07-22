import os
import argparse
import json
import datetime
import numpy as np
import skimage.draw


def poly2mask(blobs, c, path_to_masks_folder, h, w, label, idx):
    """
    https://medium.com/@dataturks/converting-polygon-bounded-boxes-in-the-dataturks-json-format-to-mask-images-f747b7ba921c
    """
    mask = np.zeros((h, w))
    for l in blobs:
        fill_row_coords, fill_col_coords = draw.polygon(l[1], l[0], l[2])
        mask[fill_row_coords, fill_col_coords] = 1
    io.imsave(path_to_masks_folder + "/" + str(c) +
              "_" + label + "_" + str(idx) + ".png", mask)


def load_image(annotation, image):
    """
    Load image and annotations
    annotation: annotation json file from supervise.ly
    image: image file
    """
    annot = json.load(open(annotation))
    # annotation = [ a for a in annotations if a['objects'] ] # remove annotations without annotated objects
    height, width = int(annot["size"]["height"]), int(annot["size"]["width"])
    img_objects = {}
    objects = annot["objects"]
    for obj in objects:
        classTitle = obj["classTitle"]
        points = obj["points"]["exterior"]
        x_coord = []
        y_coord = []
        # list of poinst (x and y coordinates), e.g. [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...]
        for sp in obj["points"]["exterior"]:
            x_coord.append(p[0])
            y_coord.append(p[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Display images annotation mask')
    parser.add_argument(
        '-i', '--image', help='image to select', required=True)
    parser.add_argument('-a', '--annotation',
                        help='image annotation', required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print("Image does not exist")
        exit()

    if not os.path.isfile(args.annotation):
        print("Annotation doesn't exist")
        exit()

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
             # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    load_image(args.annotation, args.image)