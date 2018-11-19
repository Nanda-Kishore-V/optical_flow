import cv2
import numpy as np
from scipy import signal

from estimateFeatureTranslation import estimateFeatureTranslation

def return_derivatives(img):
    # Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    # Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    dx = np.array([1, -1]).reshape(1,-1)
    dy = np.array([1, -1]).reshape(-1,1)
    Ix = signal.convolve2d(img, dx)
    Iy = signal.convolve2d(img, dy)
    return Ix, Iy

def estimateAllTranslation(startXs, startYs, img1, img2):
    newXs = np.empty_like(startXs)
    newYs = np.empty_like(startYs)
    for idx1, (colX, colY) in enumerate(zip(startXs.T, startYs.T)):
        for idx2, (startX, startY) in enumerate(zip(colX, colY)):
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            Ix, Iy = return_derivatives(gray1)
            if startX >= 0 and startY >= 0:
                newX, newY = estimateFeatureTranslation(startX, startY, Ix, Iy, gray1, gray2)
                newXs[idx2,idx1] = newX
                newYs[idx2,idx1] = newY 
            else:
                newXs[idx2, idx1] = -1
                newYs[idx2, idx1] = -1

    return newXs, newYs
