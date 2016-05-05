from numpy.random import RandomState
import numpy as np
from lasagnekit.easy import iterate_minibatches

class SubSampled(object):

    def __init__(self, dataset, nb, random_state=2, mode='random', shuffle=True):
        self.dataset = dataset
        self.nb = nb
        self.rng = RandomState(random_state)
        self.mode = mode
        self.shuffle = shuffle
        self.next_batch_iter = None

    def load(self):
        self.dataset.load()
        if self.mode == 'random':
            indices = self.rng.randint(0, len(self.dataset.X), size=self.nb)
        elif self.mode == 'batch':
            try:
                indices = next(self.next_batch_iter)
            except Exception:
                self.next_batch_iter = iterate_minibatches(self.dataset.X.shape[0], self.nb, shuffle=self.shuffle)
                indices = next(self.next_batch_iter)
        self.X = self.dataset.X[indices]
        if hasattr(self.dataset, "y"):
            self.y = [self.dataset.y[ind] for ind in indices]
        if hasattr(self.dataset, "img_dim"):
            self.img_dim = self.dataset.img_dim
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim
        if hasattr(self.dataset, "y_raw"):
            self.y_raw = [self.dataset.y_raw[ind] for ind in indices]
