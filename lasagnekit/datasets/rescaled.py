from skimage.transform import resize
from lasagnekit.easy import linearize
import numpy as np


class Rescaled(object):

    def __init__(self, dataset, size, reshape=True):
        self.dataset = dataset
        self.size = size
        self.reshape = reshape

    def load(self):
        self.dataset.load()
        w, h = self.size
        if len(self.dataset.img_dim) == 2:
            real_w, real_h = self.dataset.img_dim
        elif len(self.dataset.img_dim) == 3:
            real_w, real_h, color = self.dataset.img_dim

        X_orig = self.dataset.X
        if (w, h) != (real_w, real_h):
            orig_shape = tuple([X_orig.shape[0]] + list(self.dataset.img_dim))
            X_orig_reshaped = X_orig.reshape(orig_shape)
            if len(self.dataset.img_dim) == 3:
                shape = tuple([X_orig.shape[0], w, h, color])
            elif len(self.dataset.img_dim) == 2:
                shape = tuple([X_orig.shape[0], w, h])
            X_b = np.empty(shape)

            if len(self.dataset.img_dim) == 3:
                shape_resize = (w, h)
            elif len(self.dataset.img_dim) == 2:
                shape_resize = (w, h)
            for i in range(X_b.shape[0]):
                X_b[i] = resize(
                    X_orig_reshaped[i], shape_resize, preserve_range=True)
            X_b = X_b.astype(np.float32)
            self.X = X_b
            self.img_dim = self.X.shape[1:]
        else:
            self.X = X_orig
            self.img_dim = self.dataset.img_dim
        if self.reshape:
            self.X = linearize(self.X)
        if hasattr(self.dataset, "y"):
            self.y = self.dataset.y
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim
        if hasattr(self.dataset, "y_raw"):
            self.y_raw = self.dataset.y_raw
