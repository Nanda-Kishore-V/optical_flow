#Input: bb_corners: (4x2 numpy array) XY_TOP_LEFT, XY_TOP_RIGHT, XY_BOTTOM_LEFT, XY_BOTTOM_RIGHT
#Input: image (nxmx3 numpy array)
#Output: bb_img: image with bounding box

import cv2

def draw_bounding_box(bb_corners, img):
    bb_img = img
    x, y, w, h = cv2.boundingRect(bb_corners)
    cv2.rectangle(bb_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return bb_img