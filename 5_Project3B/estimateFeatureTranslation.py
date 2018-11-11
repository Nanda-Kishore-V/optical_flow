import cv2
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def return_derivatives(img):
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return Ix, Iy

def estimateFeatureTranslation(startX, startY, img1, img2):
    img1_cpy = img1.copy()
    img2_cpy = img2.copy()

    Ix, Iy = return_derivatives(img1_cpy)
    It = img2-img1
    Ix_func = RegularGridInterpolator((range(Ix.shape[0]), range(Ix.shape[1])), Ix, bounds_error = False, fill_value = None)
    Iy_func = RegularGridInterpolator((range(Iy.shape[0]), range(Iy.shape[1])), Iy, bounds_error = False, fill_value = None)
    It_func = RegularGridInterpolator((range(It.shape[0]), range(It.shape[1])), It, bounds_error = False, fill_value = None)

    i,j = np.meshgrid(np.arange(startY-4.5,startY+4.6,1), np.arange(startX-4.5,startX+4.6,1),indexing = 'ij')
    window_indices = np.dstack((i,j))
    Ix_window = Ix_func(window_indices)
    Iy_window = Iy_func(window_indices)
    It_window = It_func(window_indices)

    A = np.zeros((2,2))
    A[0,0] = np.sum(Ix_window*Ix_window)
    A[0,1] = np.sum(Ix_window*Iy_window)
    A[1,0] = np.sum(Iy_window*Ix_window)
    A[1,1] = np.sum(Iy_window*Iy_window)
    b = np.zeros((2,1))
    b[0,0] = -np.sum(Ix_window*It_window)
    b[1,0] = -np.sum(Iy_window*It_window)

    uv = np.linalg.solve(A,b)
    newX = startX + uv[0,:]
    newY = startY + uv[1,:]
    return newX, newY
