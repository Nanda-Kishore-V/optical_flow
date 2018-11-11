import cv2
from scipy import signal

from scipy.interpolate import RegularGridInterpolator

def blur(img):
    return cv2.blur(img, (5,5))

def return_derivatives(img):
    gaussianKernel = GaussianPDF_2D(0,1,7,7)
    dx = np.array([1,-1]).reshape(1,2)
    Gx = signal.convolve2d(gaussianKernel, dx, mode = 'same')
    Gy = signal.convolve2d(gaussianKernel, dy, mode = 'same')
    Ix = signal.convolve2d(I_gray, Gx, mode = 'same')
    Iy = signal.convolve2d(I_gray, Gy, mode = 'same')
    return Ix, Iy

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
    img1_cpy = img1.copy()
    img1_blur = blur(img1_cpy)
    img2_cpy = img2.copy()
    img2__blur = blur(img2_cpy)

    Ix, Iy = return_derivatives(img1_blur)

    Ix_func = RegularGridInterpolator((range(Ix.shape[0]),range(Ix.shape[1])), Ix, bounds_error=False,fill_value=None)
    Iy_func = RegularGridInterpolator((range(Iy.shape[0]),range(Iy.shape[1])), Iy, bounds_error=False,fill_value=None)
    It = img2-img1
    
    i,j = np.meshgrid(np.arange(startX-4.5,startX+4.6,1), np.arange(startY-4.5,startY+4.6,1),indexing = 'ij')
