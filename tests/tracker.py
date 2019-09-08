import cv2
import sys
from random import randint


# In tracking, our goal is to find an object in the current frame given we have tracked the object successfully in all ( or nearly all ) previous frames
# https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

if __name__ == "__main__":

    vid = cv2.VideoCapture("../dataset/test.mp4")
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # Multi tracker init
    trackers = cv2.MultiTracker_create()

    while True:
        return_value, frame = vid.read()

        if not return_value:
            break

        rects = []
        
        # get updated location of objects in subsequent frames
        success, boxes = trackers.update(frame)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'p' key is selected, we select a bounding box to track
        if key == ord("p"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            box = cv2.selectROI("Frame", frame, fromCenter=False,
                showCrosshair=True)
    
            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
            tracker = cv2.TrackerCSRT_create()
            trackers.add(tracker, frame, box)
        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()