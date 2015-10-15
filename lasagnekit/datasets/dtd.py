import lasagnekit
import os
import re
from glob import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np

class DTD(object):
    category_names = [
        re.match(os.getenv("DATA_PATH") + "/textures-dtd/images/(.*)", dirname).groups(1)[0]
        for dirname in glob(os.getenv("DATA_PATH") + "/textures-dtd/images/*")
    ]
    all_images = list(glob(os.getenv("DATA_PATH") + "/textures-dtd/images/**/*.jpg"))

    def __init__(self, examples_filter=None,
                       w=200, h=200):
        if examples_filter is None:
            examples_filter = np.arange(5640)
        self.examples_filter = examples_filter
        self.w = w
        self.h = h

    def load(self):
        X = []
        y = []

        images = np.array(self.all_images)[self.examples_filter]
        for image in images:
            category_name = re.match(os.getenv("DATA_PATH") + "/textures-dtd/images/(.*)/.*jpg", image).groups(1)[0]
            image = imread(image)
            image = resize(image, (self.h, self.w))
            X.append(image)
            y.append(category_name)
        X = np.array(X).astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        X = lasagnekit.easy.linearize(X)
        y = np.array(y)


        self.img_dim = (3, self.h, self.w)
        self.X = X
        self.y = y
