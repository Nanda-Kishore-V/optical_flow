import cv2
import os

import numpy as np
import scipy.misc

from draw_bounding_box import draw_bounding_box
from get_features import get_features
from estimateFeatureTranslation import estimateFeatureTranslation
from applyGeometricTransformation import applyGeometricTransformation

def objectTracking():
    img1 = np.array([[0, 0,  0,  0, 0,  0, 0, 0, 0],
                     [0, 2,  4,  5, 4,  2, 0, 0, 0],
                     [0, 4,  9, 12, 9,  4, 0, 0, 0],
                     [0, 5, 12, 15, 12, 5, 0, 0, 0],
                     [0, 4,  9, 12, 9,  4, 0, 0, 0],
                     [0, 2,  4,  5, 4,  2, 0, 0, 0],
                     [0, 0,  0,  0, 0,  0, 0, 0, 0],
                     [0, 0,  0,  0, 0,  0, 0, 0, 0],
                     [0, 0,  0,  0, 0,  0, 0, 0, 0]])
    img2 = np.array([[0, 0, 0,  0,  0, 0,  0, 0, 0],
                     [0, 0, 0,  0,  0, 0,  0, 0, 0],
                     [0, 0, 0,  0,  0, 0,  0, 0, 0],
                     [0, 0, 2,  4,  5, 4,  2, 0, 0],
                     [0, 0, 4,  9, 12, 9,  4, 0, 0],
                     [0, 0, 5, 12, 15, 12, 5, 0, 0],
                     [0, 0, 4,  9, 12, 9,  4, 0, 0],
                     [0, 0, 2,  4,  5, 4,  2, 0, 0],
                     [0, 0, 0,  0,  0, 0,  0, 0, 0]])

    newX, newY = estimateFeatureTranslation(3, 3, img1, img2)
    print(newX, newY)

if __name__ == "__main__":
    objectTracking()
