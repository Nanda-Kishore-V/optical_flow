import numpy as np
import skimage.transform as tf

from ransac import ransac

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    # NOTE: The input arrays must be atleast 2D
    N = startXs.shape[0]
    F = startXs.shape[1]
    print(N)
    print(F)
    Xs = np.empty((N, F))
    Ys = np.empty((N, F))
    newbbox = np.empty((F, 4, 2))
    for i in range(F):
        Xs[:,i], Ys[:,i], newbbox[i,:,:] = ransac(startXs[:,i], startYs[:,i], newXs[:,i], newYs[:,i], bbox[i,:])
        '''
        old code - remove later
        '''
        # src = np.stack((startXs[:,i], startYs[:,i]), axis=-1)
        # dst = np.stack((newXs[:,i], newYs[:,i]), axis=-1)
        # transf = tf.SimilarityTransform()
        # transf.estimate(src, dst)
        # T = transf.params
        # coords = np.block([[startXs[:,i].reshape(1,-1)], [startYs[:,i].reshape(1,-1)], [np.ones((1, N))]])
        # new_coords = np.dot(T, coords)
        # Xs[:,i] = new_coords[0,:]
        # Ys[:,i] = new_coords[1,:]
        # errors = np.sqrt(np.sum((np.transpose(dst) - np.block([[Xs[:,i]], [Ys[:,i]]]))**2, axis=0))
        # Xs[errors > 4, i] = -1
        # Ys[errors > 4, i] = -1
        # curr_bbox = bbox[i,:]
        # newbb = np.dot(T, np.block([[np.transpose(curr_bbox)], [np.ones((1, 4))]]))
        # newbb = newbb[[0,1],:]
        # newbb = np.transpose(newbb)
        # newbbox[i,:,:] = newbb
    return Xs, Ys, newbbox

# if __name__=="__main__":
#     applyGeometricTransformation(np.array([[1], [2], [3] ,[4]]), np.array([[5], [6], [7], [8]]),
#     np.array([[9], [10], [11], [12]]), np.array([[13], [14], [15], [16]]), np.arange(8).reshape(1,4,2))
