import os
import lasagnekit
import numpy as np
from scipy.io import loadmat

class NotMNIST(object):

    def __init__(self):
        pass

    def load(self):
        data = loadmat(os.path.join(os.getenv("DATA_PATH"), "notmnist", "notMNIST_small.mat"))
        
        X = data['images'].transpose( (2, 0, 1) )
        y = data['labels']
        X = lasagnekit.easy.linearize(np.array(list(X)))
        X = X.astype(np.float32) / 255.
        self.X = X
        self.y = y
