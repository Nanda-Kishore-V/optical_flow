import cv2
import os

import numpy as np
import skvideo.io

from draw_bounding_box import draw_bounding_box
from get_features import get_features
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation

def objectTracking(filename):
    cap = cv2.VideoCapture(filename)
    img1 = None
    img2 = None
    writer = skvideo.io.FFmpegWriter('Easy.avi')
    bboxs = np.load('easy.npy')
    frame_num = 0
    while(cap.isOpened()):
        frame_num += 1
        ret, frame = cap.read()
        if not ret:
            break
        if img1 is None and img2 is None:
            img2 = frame
            img2 = cv2.GaussianBlur(img2, (7, 7), 0)
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            startYs, startXs = get_features(gray, bboxs)
            continue
        img1 = img2
        img2 = frame
        img2 = cv2.GaussianBlur(img2, (7, 7), 0)
        newXs, newYs = estimateAllTranslation(startXs, startYs, img1, img2)
        startXs, startYs, bboxs = applyGeometricTransformation(startXs, startYs, newXs, newYs, bboxs)

        bb_img = frame
        delete_mask = np.ones(bboxs.shape[0], dtype=bool)
        for idx, bbox in enumerate(bboxs):
            mask = np.logical_or(startXs[:,idx]>= frame.shape[1],startYs[:,idx]>= frame.shape[0])
            startXs[:,idx][mask] = -1
            startYs[:,idx][mask] = -1
            if (startXs[:, idx] < 0).all() and (startYs[:, idx] < 0).all():
                delete_mask[idx] = False
                continue
            bb_img = draw_bounding_box(bbox, bb_img)

        bboxs = bboxs[delete_mask,:,:]
        startXs = startXs[:,delete_mask]
        startYs = startYs[:,delete_mask]


        for idx, (x,y) in enumerate(zip(startXs, startYs)):
            for ind in range(bboxs.shape[0]):
                if x[ind]>=0 and y[ind]>=0:
                    cv2.circle(bb_img,(np.int32(x[ind]),np.int32(y[ind])),3,(0,0,255),-1)
        writer.writeFrame(bb_img[:,:,[2,1,0]])
        cv2.imshow('frame', bb_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    objectTracking('videos/Easy.mp4')
