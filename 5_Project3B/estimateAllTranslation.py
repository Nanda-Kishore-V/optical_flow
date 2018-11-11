import cv2
import numpy as np

from estimateFeatureTranslation import estimateFeatureTranslation


def estimateAllTranslation(startXs, startYs, img1, img2):
    newXs = np.empty_like(startXs)
    newYs = np.empty_like(startYs)
    for idx, (startX, startY) in enumerate(zip(startXs, startYs)):
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        newX, newY = estimateFeatureTranslation(startX[0], startY[0], gray1, gray2)
        newXs[idx] = newX
        newYs[idx] = newY
    return newXs, newYs