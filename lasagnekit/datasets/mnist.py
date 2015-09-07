import os
import lasagnekit
import numpy as np

class MNIST(object):

    def __init__(self, which='all'):
        self.which = which

    def load(self):

        if self.which != 'all':
            X, y = np.load(os.path.join(os.getenv("DATA_PATH"), "mnist", "%s.npy" % (self.which,)))
            X = list(X)
            y = list(y)
        else:
            X_train, y_train = np.load(os.path.join(os.getenv("DATA_PATH"), "mnist", "train.npy"))
            X_test, y_test   = np.load(os.path.join(os.getenv("DATA_PATH"), "mnist", "test.npy"))
            X = list(X_train) + list(X_test)
            y = list(y_train) + list(y_test)
        X = np.array(X)
        self.img_dim = X.shape[1:]
        X = lasagnekit.easy.linearize(X)
        X = X.astype(np.float32) / 255.
        y = np.array(y)[:, 0]
        self.X = X
        self.y = y
