from collections import deque
from imutils.video import VideoStream
from networktables import NetworkTables
import numpy as np
import cv2
import imutils

red_lower = (165, 180, 6)
red_upper = (180, 255, 255)

capture = cv2.VideoCapture(0)

NetworkTables.initialize(server="10.55.216.17")
sd = NetworkTables.getTable("SmartDashboard")

def send_to_robot(data):
    sd.putValue("ball_x", data[0])
    sd.putValue("ball_y", data[1])
    sd.putValue("ball_on_screen", data[2])

while True:
    ret, frame = capture.read()

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, red_lower, red_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            data = [round(x, 2), round(y, 2), True]
            send_to_robot(data)
        else:
            data = [0, 0, False]
            send_to_robot(data)

    cv2.imshow("capture", frame)
    cv2.waitKey(1)
