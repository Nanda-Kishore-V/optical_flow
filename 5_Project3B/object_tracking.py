import cv2
import os

import numpy as np
import scipy.misc

from draw_bounding_box import draw_bounding_box
from get_features import get_features
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

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
            img2 = cv2.GaussianBlur(img2, (7,7), 0)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bboxs = np.empty((2,4,2)) #hardcoded 1
            # bboxs[0] = np.array([[315, 192],[373,192],[380,244],[309,241]])
            # bboxs[0] = np.array([[262,124],[262,70],[308,70],[308,124]])
            bboxs[0] = np.array([[462,216],[500,216],[500,240],[462,240]])
            # for bbox in bbox:
            startYs, startXs = get_features(gray, bboxs[0])
            continue

        img1 = img2
        img2 = frame
        img2 = cv2.GaussianBlur(img2, (7,7), 0)
        newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)
        Xs, Ys, bbox_new = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)

        bb_img = draw_bounding_box(bbox_new[0], frame)
        Xs = np.reshape((Xs[Xs != -1]),(-1,1))
        Ys = np.reshape((Ys[Ys != -1]),(-1,1))
        startXs = Xs
        startYs = Ys

        for idx, (x,y) in enumerate(zip(Xs, Ys)):
            cv2.circle(bb_img,(x,y),3,(0,0,255),-1)

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
    objectTracking('videos/Medium.mp4')
