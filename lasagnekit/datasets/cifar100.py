import os
import lasagnekit
import numpy as np

class Cifar100(object):

    def __init__(self, which='all', coarse_label=False):
        self.which = which
        self.coarse_label = coarse_label

    def load(self):
        if self.which == 'all':
            ds = ['train', 'test']
        else:
            ds = [self.which]
        X_all = []
        y_all = []
        for w in ds:
            data = np.load(os.path.join(os.getenv("DATA_PATH"), "cifar100", "%s" % (w,)))
            X = data['data']
            if self.coarse_label is True:
                y = data['coarse_labels']
            else:
                y = data['fine_labels']
            X_all.append(X)
            y_all.append(y)
        
        
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        X = lasagnekit.easy.linearize(X)
        X = X.astype(np.float32) / 255.
        y = np.array(y).astype(np.int32)
        self.X = X
        self.y = y
