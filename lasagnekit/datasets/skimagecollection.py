import numpy as np


class ImageCollection(object):

    def __init__(self, collection, indices=None, batch_size=100):
        self.collection = collection
        if indices is None:
            indices = np.arange(len(collection))
        self.indices = indices
        self.batch_pointer = 0
        self.batch_size = batch_size

    def load(self):
        nb = min(self.batch_size, len(self.collection) - self.batch_pointer)
        self.X = [self.collection[self.indices[i + self.batch_pointer]]
                  for i in range(nb)]
        self.X = np.array(self.X).astype(np.float32)
        self.batch_pointer += nb
        if self.batch_pointer >= len(self.collection):
            self.batch_pointer = 0
        self.img_dim = self.X.shape[1:]
