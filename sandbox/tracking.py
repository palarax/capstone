import numpy as np
import cv2
import sys

# used
# https://medium.com/@adamaulia/object-tracking-using-opencv-python-windows-616fb23da720

# should try
# https://github.com/apoorvavinod/Real_time_Object_detection_and_tracking/blob/master/src/main.py

# read video file
cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()


while (cap.isOpened):

    # if ret is true than no error with cap.isOpened
    ret, frame = cap.read()

    if ret == True:

        # apply background substraction
        fgmask = fgbg.apply(frame)

        (im2, contours, hierarchy) = cv2.findContours(
                fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # looping for contours
        for c in contours:
            if cv2.contourArea(c) < 500:
                continue

            # get bounding box from countour
            (x, y, w, h) = cv2.boundingRect(c)

            # draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('foreground and background', fgmask)
        cv2.imshow('rgb', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
