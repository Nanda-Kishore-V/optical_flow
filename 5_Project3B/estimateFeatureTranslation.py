import cv2
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
import numpy as np

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    img1_cpy = img1.copy()
    img2_cpy = img2.copy()
    img1_func = RegularGridInterpolator((range(img1_cpy.shape[0]), range(img1_cpy.shape[1])), img1_cpy, bounds_error = False, fill_value = None)
    img2_func = RegularGridInterpolator((range(img2_cpy.shape[0]), range(img2_cpy.shape[1])), img2_cpy, bounds_error = False, fill_value = None)

    Ix_func = RegularGridInterpolator((range(Ix.shape[0]), range(Ix.shape[1])), Ix, bounds_error = False, fill_value = None)
    Iy_func = RegularGridInterpolator((range(Iy.shape[0]), range(Iy.shape[1])), Iy, bounds_error = False, fill_value = None)

    uv = np.array([[0],[0]])
    newX, newY = startX.copy(), startY.copy()
    iterations = 0
    while iterations < 10:
        i,j = np.meshgrid(np.arange(startY-4.5,startY+4.6,1), np.arange(startX-4.5,startX+4.6,1),indexing = 'ij')
        window_indices = np.dstack((i,j))
        Ix_window = Ix_func(window_indices)
        Iy_window = Iy_func(window_indices)

        shifted_i, shifted_j = np.meshgrid(np.arange(newY-4.5,newY+4.6,1), np.arange(newX-4.5,newX+4.6,1),indexing = 'ij')
        shifted_window_indices = np.dstack((shifted_i, shifted_j))
        It_window = img2_func(shifted_window_indices)-img1_func(window_indices)

        zero_window = np.zeros((10,10))
        A = np.zeros((2,2))
        A[0,0] = np.sum(Ix_window*Ix_window)
        A[0,1] = np.sum(Ix_window*Iy_window)
        A[1,0] = np.sum(Iy_window*Ix_window)
        A[1,1] = np.sum(Iy_window*Iy_window)
        b = np.zeros((2,1))
        b[0,0] = -np.sum(Ix_window*It_window)
        b[1,0] = -np.sum(Iy_window*It_window)
        try:
            uv = np.linalg.solve(A,b)
            #uv = np.dot(np.dot(np.linalg.inv(np.dot(A.T,A)),A.T),b)
        except:
            if newX>=img1.shape[1]:
                newX = startX
            if newY>=img1.shape[0]:
                newY = startY 
            print(newX, newY)
            break
        newX = newX + uv[0,:]
        newY = newY + uv[1,:]
        if np.allclose(uv, np.array([[0],[0]]), rtol=0.05, atol = 0.05):
            break
        iterations += 1

    return newX, newY
