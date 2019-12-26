#!/usr/bin/env python
import sys

sys.path.append("../modules/")
import cv2
import numpy as np
from matplotlib import pyplot as plt
from detectors import inrange
import os
from image_processing import to_three
#import rospy
#from sensor_msgs.msg import Image, CompressedImage
#from std_msgs.msg import String
#from geometry_msgs.msg import Point, Polygon
#from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

sys.path.append("../modules/")

from input_output import Source
import detectors

def nothing (x):
    pass

#source = Source ("/Users/elijah/Downloads/Q20cCyUTBeY.jpg")
source = Source ("/Users/elijah/Dropbox/Programming/detectors/images/two_objects.jpg")

low_th  = (57, 150, 110)
high_th = (67, 160, 120)

inrange_filter = inrange (low_th, high_th)

cv2.namedWindow ('Colorbars')

cv2.createTrackbar ("l1", "Colorbars",   0, 255, nothing)
cv2.createTrackbar ("h1", "Colorbars", 255, 255, nothing)
cv2.createTrackbar ("l2", "Colorbars",   0, 255, nothing)
cv2.createTrackbar ("h2", "Colorbars", 255, 255, nothing)
cv2.createTrackbar ("l3", "Colorbars",   0, 255, nothing)
cv2.createTrackbar ("h3", "Colorbars", 255, 255, nothing)

#def callback(image_msg):

while (True):
    #frame = CvBridge().imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    
    #print(frame.shape)
    
    frame = source.get_frame ()

    frame = cv2.cvtColor (frame, cv2.COLOR_YCrCb2RGB)
    frame = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)

    l1 = cv2.getTrackbarPos ("l1", "Colorbars")
    h1 = cv2.getTrackbarPos ("h1", "Colorbars")
    l2 = cv2.getTrackbarPos ("l2", "Colorbars")
    h2 = cv2.getTrackbarPos ("h2", "Colorbars")
    l3 = cv2.getTrackbarPos ("l3", "Colorbars")
    h3 = cv2.getTrackbarPos ("h3", "Colorbars")

    low_th  = (l1, l2, l3)
    high_th = (h1, h2, h3)
    print("@")
    inrange_filter.set_ths (low_th, high_th)

    mask = inrange_filter.apply (frame)
    print("&")
    enlighted = frame.copy ()
    enlighted [:, :, 0] = np.array (np.add (enlighted [:, :, 0], np.multiply (mask, 0.5)), dtype = np.uint8)

    #cv2.imshow ("enlighted", enlighted)
    #cv2.imshow ("mask", mask)
    cv2.waitKey(2)
    result = np.concatenate ((enlighted, to_three (mask)), axis = 1)
    cv2.waitKey(2)
    cv2.imshow ("result", result)
    os.system ('clear')    
    print (low_th, high_th)

