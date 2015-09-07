
import numpy as np
import os
class ILC(object):

    def __init__(self, name="images", binarize=True):
        self.name = name
        self.binarize = binarize
        self.X = None
        self.y = None

    def load(self):
        base_path = os.path.join(os.getenv("DATA_PATH"), "ilc")
        set_file = "%s.npy" % (self.name,)
        full_path = os.path.join(base_path, set_file)
        X = np.load(full_path)
        N = np.prod( X.shape[1:] )
        X = X.reshape(  (X.shape[0], N))
        if self.binarize:
            X = 1.*(X > 0)
        self.X = X
        self.y = None
