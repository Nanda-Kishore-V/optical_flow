#In this section, you will identify features within the bounding box for each object using Harris corners or Shi-Tomasi
#features. We recommend you to use corner_harris or corner_shi_tomasi in Python. Good features to track are the ones
#whose motion can be estimated reliably. You can perform some kind of thresholding or local maxima suppression if the
#number of features obtained are too large.
#Complete the following function:
#[x,y]=getFeatures(img,bbox)
# (INPUT) img: H W matrix representing the grayscale input image
# (INPUT) bbox: F 42 matrix representing the four corners of the bounding box where F is the number of
#objects you would like to track
# (OUTPUT) x: N F matrix representing the N row coordinates of the features across F objects
# (OUTPUT) y: N F matrix representing the N column coordinates of the features across F objects
#Here, N is the maximum number of features across all the bounding boxes. You can fill in the missing rows with either 0
#or -1 or any other number that you prefer.

def get_features(img,bbox):
    #harris corner detect from skimage
    #harris extract
    return x,y