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
        if img1 is None and img2 is None:
            img1 = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bbox = np.array([[262,124],[262,70],[308,70],[308,124]])
            i_fps, j_fps = get_features(gray, bbox)
            fp = (j_fps[11],i_fps[11])
            continue
        img2 = img1
        img1 = frame
        if ret == False:
            break
        #scipy.misc.imsave('outfile.jpg', frame) for saving individual frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #for color conversion
        #gray = cv2.resize(gray, (240, 320)) #for gray scale
        #call function here #implement our logic here
        #corner_shi_tomasi(frame)
        bbox = np.array([[262,124],[262,70],[308,70],[308,124]])
        bb_img = draw_bounding_box(bbox, frame)
        
        #bb_img[i_fps,j_fps,0] = 0
        #bb_img[i_fps,j_fps,1] = 0
        #bb_img[i_fps,j_fps,2] = 255
        
        
        newX, newY = estimateFeatureTranslation(fp[0], fp[1],img1, img2)
        newX = int(newX)
        newY = int(newY)
        #bb_img[newY, newX,0] = 0
        #bb_img[newY, newX,1] = 0
        #bb_img[newY, newX,2] = 255
        cv2.circle(bb_img,(newX,newY),5,(0,0,255),-1)
        fp = (newX,newY)
        print(newX,newY)
        cv2.imshow('frame', bb_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    videos_dir = os.fsencode('videos')
    for file in os.listdir(videos_dir):
        test_video = os.fsdecode(file)
        objectTracking('videos/'+ test_video)
