import os
import numpy as np
import lasagnekit
from skimage.color import rgb2gray

class Insects(object):

    def __init__(self, grayscale=False):
        self.grayscale = grayscale

    def load(self): 
        data = np.load(os.path.join(os.getenv("DATA_PATH"), "insects", "data_64x64.npy.npz"))
        X = data['X']
        if self.grayscale is True:
            for i in xrange(X.shape[0]):
                X[i, :, :, 0] = rgb2gray(X[i])
            X = X[:, :, :, 0]
        shape = X.shape[1:]
        X = lasagnekit.easy.linearize(X).astype(np.float32) / 255.
        y = data['y']

        labels = list(set(y.tolist()))
        mapping = {}
        for i, l in enumerate(labels):
            mapping[l] = i
        for i in xrange(y.shape[0]):
            y[i] = mapping[y[i]]

        self.X = X
        self.y = y
        self.img_dim = shape
