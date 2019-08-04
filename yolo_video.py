#! /usr/bin/env python

import argparse
import os
import numpy as np
import json
import logging

# KERAS
import tensorflow as tf
import keras.backend as K

from yolo3.yolo import YOLO, detect_video

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open("bbox_transform/"+img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
            #opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
            #cv2.imwrite('pictures/test_result.png',opencvImage)
            #cv2.imshow("Output", opencvImage)
    yolo.close_session()


# def _main_(args):
#     config_path  = args.conf
#     input_path   = args.input
#     output_path  = args.output

#     with open(config_path) as config_buffer:    
#         config = json.load(config_buffer)

#     makedirs(output_path)

          

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-c', '--conf', help='path to configuration file')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    
    args = argparser.parse_args()
    _main_(args)
