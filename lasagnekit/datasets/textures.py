
import lasagnekit

import os
import numpy as np

class Textures(object):

    def __init__(self):
        pass

    def load(self):

        X = np.load(os.path.join(os.getenv("DATA_PATH"), "textures", "textures.npy"))
        self.img_dim = X.shape[1:]
        X = lasagnekit.easy.linearize(X)
        self.X = X
