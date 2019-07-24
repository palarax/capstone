# rename img files to #####.jpg format

import os
import argparse
import glob
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PNG to JPG converter')
    parser.add_argument('-i', '--counter', type = int,
                        help='starting image number', required=True)
    parser.add_argument('-l', '--label',
                        help='image label type', required=True)
    parser.add_argument('-d', '--directory',
                        help='directory with images', required=True)
    parser.add_argument(
        '-o', '--output', help='Output directory', default='./output')
    args = parser.parse_args()

    counter = args.counter # where should the counter start from
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # rename and move all JPEG images
    for img in [f for f in glob.glob(args.directory+"/*") if Image.open(f).format == 'JPEG']:
        os.rename(img, os.path.join(args.output, "{}_{}.jpg".format(args.label, counter)))
        counter += 1