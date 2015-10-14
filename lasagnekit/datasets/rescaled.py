from skimage.transform import resize
from lasagnekit.easy import linearize
import os
import numpy as np

class Rescaled(object):

    def __init__(self, dataset, size, cache=False):
        self.dataset = dataset 
        self.size = size
        self.cache = cache

    def load(self):
        self.dataset.load()
        w, h = self.size
        if len(self.dataset.img_dim) == 2:
            real_w, real_h = self.dataset.img_dim
        elif len(self.dataset.img_dim) == 3:
            color, real_w, real_h = self.dataset.img_dim

        X_orig = self.dataset.X

        if (w, h) != (real_w, real_h):
            name = "{0}-{1}x{2}.npy".format(str(self.dataset.__class__), w, h)
            if os.path.exists(name) and self.cache == True:
                self.X =  np.load(name)
                self.y = self.dataset.y
            else:
                orig_shape = tuple([X_orig.shape[0]] + list(self.dataset.img_dim))
                X_orig_reshaped = X_orig.reshape(orig_shape)
                if len(self.dataset.img_dim) == 3:
                    shape = tuple([X_orig.shape[0], self.dataset.img_dim[0], w, h])
                elif len(self.dataset.img_dim) == 2:
                    shape = tuple([X_orig.shape[0], w, h])
                X_b = np.empty(shape)
                
                if len(self.dataset.img_dim) == 3:
                    shape_resize = [orig_shape[1], w, h]
                elif len(self.dataset.img_dim) == 2:
                    shape_resize = (w, h)
                for i in range(X_b.shape[0]):
                    X_b[i] = resize(X_orig_reshaped[i], shape_resize, preserve_range=True)
                X_b = X_b.astype(np.float32)
                if self.cache == True:
                    np.save(name, X_b)            
                self.X = X_b
                self.y = self.dataset.y
        else:
            self.X = X_orig
            self.y = self.dataset.y
        
        self.img_dim = self.X.shape[1:]
        self.X = linearize(self.X)
