import numpy as np
import skimage.transform as tf
import random

def ransac(startXs, startYs, newXs, newYs, bbox):
    max_inliers = -np.inf
    transformation = None

    N = startXs.shape[0]
    print(startXs.shape[0])
    print(startYs.shape[0])
    coords = np.block([[startXs.reshape(1,-1)], [startYs.reshape(1,-1)], [np.ones((1, N))]])
    src = np.stack((startXs, startYs), axis=-1)
    dst = np.stack((newXs, newYs), axis=-1)

    thresh = 1

    for _ in range(100):
        # indices = np.random.randint(0, N, size=3)
        if N < 3:
            print('No points to perform RANSAC on.')
            return newXs, newYs, bbox
        indices = random.sample(range(N), 3)
        src_i = np.stack((startXs[indices], startYs[indices]), axis=-1)
        dst_i = np.stack((newXs[indices], newYs[indices]), axis=-1)

        transf = tf.SimilarityTransform()
        transf.estimate(src_i, dst_i)
        T = np.array(transf.params)

        new_coords = np.dot(T, coords)
        Xs = new_coords[0,:]
        Ys = new_coords[1,:]

        errors = np.sqrt(np.sum((np.transpose(dst) - np.block([[Xs], [Ys]]))**2, axis=0))
        # if np.isnan(errors).any():
            # print(indices)
            # print(src_i)
            # print(dst_i)
            # print(errors)
        inliers = errors[errors < thresh].size
        if max_inliers < inliers:
            max_inliers = inliers
            transformation = T

    new_coords = np.dot(transformation, coords)
    Xs = new_coords[0,:]
    Ys = new_coords[1,:]

    errors = np.sqrt(np.sum((np.transpose(dst) - np.block([[Xs], [Ys]]))**2, axis=0))
    Xs[errors > thresh] = -1
    Ys[errors > thresh] = -1
    bb = np.dot(transformation, np.block([[np.transpose(bbox)], [np.ones((1, 4))]]))
    bb = bb[[0,1],:]
    bb = np.transpose(bb)
    return Xs, Ys, bb
