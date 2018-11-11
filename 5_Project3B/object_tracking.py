import cv2
import os

import numpy as np
import scipy.misc

from draw_bounding_box import draw_bounding_box
from get_features import get_features
from estimateFeatureTranslation import estimateFeatureTranslation

def objectTracking(filename):
    cap = cv2.VideoCapture(filename)
    img1 = None
    img2 = None
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        if img1 is None and img2 is None:
            img2 = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = np.array([[262,124],[262,70],[308,70],[308,124]])
            i_fps, j_fps = get_features(gray, bbox)
            fp = [j_fps[11], i_fps[11]]
            continue
        img1 = img2
        img2 = frame
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #for color conversion
        bbox = np.array([[262,124],[262,70],[308,70],[308,124]])
        bb_img = draw_bounding_box(bbox, frame)

        newX, newY = estimateFeatureTranslation(fp[0], fp[1], gray1, gray2)
        cv2.circle(bb_img,(newX,newY),5,(0,0,255),-1)
        fp = (newX, newY)
        cv2.imshow('frame', bb_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # videos_dir = os.fsencode('videos')
    # for file in os.listdir(videos_dir):
    #     test_video = os.fsdecode(file)
    #     objectTracking('videos/'+ test_video)
    objectTracking('videos/Easy.mp4')
