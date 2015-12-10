import os
from numpy.random import RandomState
from skimage.io import imread
import numpy as np


class ImageNet(object):

    def __init__(self, mode="random", random_state=2, nb=100):
        self.mode = mode
        self.rng = RandomState(random_state)
        self.nb = nb

    def load(self):
        path = os.path.join(os.getenv("DATA_PATH"), "imagenet",
                            "imagenet_downloader")
        dirs = os.path.listdir(path)
        dirs = filter(lambda d: os.path.isdir(d), dirs)
        dirs = self.rng.choice(dirs, size=self.nb)
        X = []
        y = []
        for d in dirs:
            filenames = os.path.listdir(path + "/" + d)
            filename = self.rng(filenames)
            x = imread(path + "/" + d + "/" + filename).tolist()
            X.append(x)
            y.append(d)
        X = np.array(X).astype(np.float32)
        self.X = X
        self.y = y
