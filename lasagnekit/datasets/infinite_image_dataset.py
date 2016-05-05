import numpy as np
from realtime_augmentation import fast_warp, random_perturbation_transform


def Transform(X, rng,
              zoom_range=None,
              rotation_range=None,
              shear_range=None, translation_range=None, do_flip=False):
    if zoom_range is None:
        zoom_range = (1.0, 1.0)
    if rotation_range is None:
        rotation_range = (0, 0)
    if shear_range is None:
        shear_range = (0, 0)
    if translation_range is None:
        translation_range = (0, 0)
    h = X.shape[1]
    w = X.shape[2]
    X_trans = np.zeros(X.shape, dtype="float32")
    transf_list = []
    for i in range(X.shape[0]):
        transf = random_perturbation_transform(
                    rng,
                    zoom_range,
                    rotation_range,
                    shear_range,
                    translation_range, do_flip=do_flip,
                    w=w, h=h)
        X_trans[i] = fast_warp(
            X[i], transf,
            output_shape=X.shape[1:], mode="constant")
        transf_list.append(transf)

    return X_trans, transf_list


class InfiniteImageDataset(object):
    online = True

    def __init__(self,
                 dataset,
                 rng=np.random,
                 **params):
        self.dataset = dataset
        self.params = params
        self.rng = rng

    def load(self):
        self.dataset.load()
        samples = self.dataset.X.reshape((self.dataset.X.shape[0],) + self.dataset.img_dim)
        color_first = (len(samples.shape) == 4 and samples.shape[1] == 3)
        if color_first:
            samples = samples.transpose((0, 2, 3, 1))
        self.X, self.X_params = Transform(samples, self.rng, **self.params)
        self.X = self.X.transpose((0, 3, 1, 2))
        self.X = self.X.reshape((self.X.shape[0], -1))
        if hasattr(self.dataset, "y"):
            self.y = self.dataset.y
        if hasattr(self.dataset, "img_dim"):
            self.img_dim = self.dataset.img_dim
        if hasattr(self.dataset, "output_dim"):
            self.output_dim = self.dataset.output_dim
        if hasattr(self.dataset, "y_raw"):
            self.y_raw = self.dataset.y_raw
