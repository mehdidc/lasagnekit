import os
from numpy.random import RandomState
from skimage.io import imread
import numpy as np
from skimage.transform import resize


class ImageCollection(object):

    online = True

    def __init__(self, mode="random", random_state=2, nb=100,
                 size=(224, 224), crop=False,
                 folder=None,
                 filename_to_label=None,
                 process_dirs=None,
                 verbose=0):
        if not hasattr(self, "folder"):
            assert folder is not None
            self.folder = folder
        if not hasattr(self, "filename_to_label"):
            if filename_to_label is None:
                def filename_to_label(directory, filename):
                    return hash(directory)
                self.filename_to_label = filename_to_label
        if not hasattr(self, "process_dirs"):
            if process_dirs is None:
                def process_dirs(dirs):
                    return filter(lambda d: os.path.isdir(d), dirs)
            self.process_dirs = process_dirs
        path = os.path.join(os.getenv("DATA_PATH"), self.folder)
        all_dirs = map(lambda d: path + "/" + d, os.listdir(path))
        self.all_dirs = self.process_dirs(all_dirs)

        self.mode = mode
        self.rng = RandomState(random_state)
        self.nb = nb
        self.size = size
        self.crop = crop
        self.verbose = verbose

        if size is not None:
            self.img_dim = (size[1], size[0], 3)
        else:
            self.img_dim = None

    def load(self):
        X = []
        y = []
        while len(X) < self.nb:
            d = self.rng.choice(self.all_dirs)
            filenames = os.listdir(d)
            filename = self.rng.choice(filenames)
            try:
                x = imread(d + "/" + filename)
                h, w = x.shape[0:2]
                if self.crop:
                    if h >= self.size[0]:
                        a = (h - self.size[0]) / 2
                        b = h - self.size[0] - a
                        x = x[a:-b]
                    if w >= self.size[1]:
                        a = (w - self.size[1]) / 2
                        b = w - self.size[1] - a
                        x = x[:, a:-b]
                    x = resize(x, self.size)
                else:
                    if self.size is not None:
                        x = resize(x, self.size)

            except Exception as ex:
                if self.verbose > 0:
                    print("Exception when processing {} : {}".format(filename, repr(ex)))
                continue
            if len(x.shape) == 2:
                x = x[:, :, None] * np.ones((1, 1, 3))
            if len(x.shape) == 3 and x.shape[-1] == 4:
                x = x[:, :, 0:3]
            if len(x.shape) == 3 and x.shape[-1] > 4:
                # there is an image with shape[2]=90, wtf?
                continue
            X.append(x)
            l = self.filename_to_label(d, filename)
            y.append(l)
        X = np.array(X).astype(np.float32)

        if self.img_dim is None:
            self.img_dim = X.shape[1:]
        self.X = X
        self.y = y
