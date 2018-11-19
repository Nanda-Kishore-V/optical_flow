import numpy as np
import skimage.transform as tf

from ransac import ransac

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
    # NOTE: The input arrays must be atleast 2D
    N = startXs.shape[0]
    F = startXs.shape[1]
    print(N)
    print(F)
    Xs = -np.ones((N, F))
    Ys = -np.ones((N, F))
    newbbox = np.empty((F, 4, 2))
    print(startXs)
    print(newXs)
    for i in range(F):
        mod_startX = startXs[:,i][startXs[:,i] < 0]
        mod_startY = startYs[:,i][startYs[:,i] < 0]
        mod_newX = newXs[:,i][newXs[:,i] < 0]
        mod_newY = newYs[:,i][newYs[:,i] < 0]
        x, y, newbbox[i,:,:] = ransac(mod_startX, mod_startY, mod_newX, mod_newY, bbox[i,:])
        Xs[:x.size, i], Ys[:y.size, i] = x, y
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
