from skimage.transform import resize
from lasagnekit.easy import linearize
import os
import numpy as np
import hashlib
import cPickle as pickle


class Cached(object):
    dirname = os.path.join(os.getenv("DATA_PATH"), "cache")

    def __init__(self, dataset):
        self.dataset = dataset

    def load(self):

        params = pickle.dumps(self.dataset.__dict__)
        m = hashlib.md5()
        m.update(params)
        filename = m.hexdigest()
        filename = os.path.join(self.dirname, filename + ".pkl")

        if os.path.exists(filename):
            data = pickle.load(open(filename))
            self.dataset.__dict__.update(data)
        else:
            self.dataset.load()
            fd = open(filename, "w")
            pickle.dump(self.dataset.__dict__, fd)
            fd.close()

        if hasattr(self.dataset, "X"):
            self.X = self.dataset.X
        if hasattr(self.dataset, "y"):
            self.y = self.dataset.y
        if hasattr(self.dataset, "y_raw"):
            self.y_raw = self.dataset.y_raw
        if hasattr(self.dataset, "img_dim"):
            self.img_dim = self.dataset.img_dim
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim
