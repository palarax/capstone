import os
import glob
import argparse
from PIL import Image


def convert(img, outDir):
    '''
    Converts PNG image to JPG format
    '''
    im = Image.open(img)
    print("Converting file: {}".format(img))
    try:
        rgb_im = im.convert('RGB')
    except Warning:
        rgb_im = im.convert('RGBA')
    rgb_im.save(os.path.join(outDir, os.path.basename(img)[:-4]+'.jpg'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PNG to JPG converter')
    parser.add_argument(
        '-i', '--image', help='Image to convert', required=False)
    parser.add_argument('-d', '--directory',
                        help='directory with images', required=False)
    parser.add_argument(
        '-o', '--output', help='Output directory', default='./output')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    if (not args.image) and (not args.directory):
        parser.error('Provide Either image or directory with images')
        exit()

    if args.image:
        if os.path.isfile(args.image):
            convert(args.image, args.outDir)

    if args.directory:
        if os.path.isdir(args.directory):
            for img in glob.glob(args.directory+"/*"):
                # if png then convert, if jpg then move to directory
                if Image.open(img).format == 'PNG':
                    convert(img, args.output)
                elif Image.open(img).format == 'JPEG':
                    os.rename(img, os.path.join(
                        args.output, os.path.basename(img)))
