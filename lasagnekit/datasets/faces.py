import os
import lasagnekit
import numpy as np

class Faces(object):

    def __init__(self, dataset='faces94', grayscale=True, batches=None):
        self.dataset = dataset
        self.grayscale = grayscale
        self.batches = batches

    def load(self):

        if self.batches is None:
            ds = np.load(os.path.join(os.getenv("DATA_PATH"), "faces", self.dataset, "data.npy.npz"))
            X = ds['X_grayscale'] if self.grayscale is True else ds['X_rgb']
        else:
            X_list = []
            for b in self.batches:
                ds = np.load(os.path.join(os.getenv("DATA_PATH"), "faces", self.dataset, "data_batch_{0}.npy.npz".format(b)))
                X_b = ds['X_grayscale'] if self.grayscale is True else ds['X_rgb']
                X_list.append(X_b)
            X  = np.concatenate(X_list, axis=0)

        self.img_dim = X.shape[1:]
        X = lasagnekit.easy.linearize(X)
        self.X = X.astype(np.float32)
