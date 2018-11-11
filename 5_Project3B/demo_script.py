import cv2
import os

import numpy as np
import scipy.misc

from draw_bounding_box import draw_bounding_box
from get_features import get_features
frame = cv2.imread('outfile.jpg')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
bbox = np.array([[262,124],[262,70],[308,70],[308,124]])
bb_img = draw_bounding_box(bbox, frame)
x_fps, y_fps = get_features(gray, bbox)
bb_img = draw_bounding_box(bbox, frame)
x_fps, y_fps = get_features(gray, bbox)
bb_img[x_fps,y_fps,0] = 0
bb_img[x_fps,y_fps,1] = 0
bb_img[x_fps,y_fps,2] = 255
cv2.imshow('frame', bb_img)
if cv2.waitKey(0) & 0xff == ord('q'):
    exit()
