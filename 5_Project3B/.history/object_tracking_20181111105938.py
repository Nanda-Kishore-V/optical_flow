import cv2
import os

import numpy as np
import scipy.misc


from draw_bounding_box import draw_bounding_box
from get_features import get_features
def objectTracking(filename):
    cap = cv2.VideoCapture(filename)

    while(cap.isOpened()):
        ret, frame = cap.read()
        #scipy.misc.imsave('outfile.jpg', frame) for saving individual frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #for color conversion
        #gray = cv2.resize(gray, (240, 320)) #for gray scale
        #call function here #implement our logic here
        #corner_shi_tomasi(frame)
        corners = np.array([[262,124],[262,70],[308,70],[308,124]])
        bb_img = draw_bounding_box(corners, frame)
        x_fps, y_fps = get_features(gray, bbox)
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