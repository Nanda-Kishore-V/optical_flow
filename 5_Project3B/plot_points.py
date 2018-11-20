import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

def plot_points():
    img1 = plt.imread('obj_fps1_m.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    startXs = np.load('obj_fpsX1_m.npy')
    startYs = np.load('obj_fpsY1_m.npy')
    endXs = np.load('obj_fpsX100_m.npy')
    endYs = np.load('obj_fpsY100_m.npy')
    for idx, (x1,y1,x2,y2) in enumerate(zip(startXs, startYs, endXs,endYs)):
            for ind in range(1):
                if x1[ind]>=0 and y1[ind]>=0 and x2[ind]>=0 and y2[ind]>=0:
                    print(str(x1) + " " + str(x2) + " " + str(y1) + " " + str(y2))
                    plt.plot([x1[ind],x2[ind]],[y1[ind],y2[ind]], 'r-')
                    #cv2.circle(img1,(np.int32(x[ind]),np.int32(y[ind])),3,(0,0,255),-1)
    plt.imshow(img1)
    plt.savefig('obj_tracking_easy.jpg')
    plt.show()
    

if __name__ == "__main__":
    plot_points()