import numpy as np


class Transformed(object):

    def __init__(self, dataset, transform_func, per_example=True):
        self.dataset = dataset
        self.transform_func = transform_func
        self.per_example = per_example

    def load(self):
        self.dataset.load()
        if self.per_example:
            X = np.empty_like(self.dataset.X)
            for i in range(self.dataset.X.shape[0]):
                X[i] = self.transform_func(self.dataset.X[i])
        else:
            X = self.transform_func(self.dataset.X)
        self.X = X
        if hasattr(self.dataset, "y"):
            self.y = self.dataset.y
        if hasattr(self.dataset, "img_dim"):
            self.img_dim = self.dataset.img_dim
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim
        if hasattr(self.dataset, "y_raw"):
            self.y_raw = self.dataset.y_raw
