import numpy as np
import cv2
import imutils


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)


if __name__ == "__main__":
    # Input image
    KNOWN_DISTANCE = 100.0
    KNOWN_WIDTH = 173
    image = cv2.imread('chey_173.jpg')
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # marker = find_marker(image)
    height, width, _ = image.shape
    print(height, width)
    img = cv2.resize(image, (1200, 900))
    box = cv2.selectROI("Frame", img, fromCenter=False,showCrosshair=True)
    h = ((box[3]) * KNOWN_DISTANCE) / KNOWN_WIDTH
    print(h)
    print(box)
    
    
    # focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    # print(focalLength)

    # inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

    # print(marker)
    # box = cv2.boxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    # box = np.int0(box)
    # cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    # cv2.putText(image, "%.2fft" % inches,
    #             (image.shape[1] - 200, image.shape[0] -
    #              20), cv2.FONT_HERSHEY_SIMPLEX,
    #             2.0, (0, 255, 0), 3)
    # img = cv2.resize(image, (600, 600))
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
