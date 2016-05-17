from numpy.random import RandomState
from manual import Manual
import numpy as np

def load_once(dataset_cls):
    class Dataset_(dataset_cls):
        def __init__(self, *args, **kwargs):
            super(Dataset_, self).__init__(*args, **kwargs)
            self.loaded = False

        def load(self):
            if self.loaded is False:
                super(Dataset_, self).load()
                self.loaded = True
    Dataset_.__name__ = dataset_cls.__name__ + "_load_once"
    return Dataset_


def split(dataset, test_size=0.5, random_state=None):
    if random_state is None:
        random_state = np.random.randint(0, 999999)
    nb = dataset.X.shape[0]
    nb_test = int(nb * test_size)
    nb_train = nb - nb_test
    rng = RandomState(random_state)
    indices = np.arange(0, nb)
    rng.shuffle(indices)
    indices_train = indices[0:nb_train]
    indices_test = indices[nb_train:]

    X = dataset.X[indices_train]
    if hasattr(dataset, 'y') and dataset.y is not None:
        y = dataset.y[indices_train]
    else:
        y = None
    dataset_train = Manual(X, y)
    if hasattr(dataset, "img_dim"):
        dataset_train.img_dim = dataset.img_dim
    if hasattr(dataset, "output_dim"):
        dataset_train.output_dim = dataset.output_dim

    X = dataset.X[indices_test]
    if hasattr(dataset, 'y') and dataset.y is not None:
        y = dataset.y[indices_test]
    else:
        y = None
    dataset_test = Manual(X, y)
    if hasattr(dataset, "img_dim"):
        dataset_test.img_dim = dataset.img_dim
    if hasattr(dataset, "output_dim"):
        dataset_test.output_dim = dataset.output_dim
    return dataset_train, dataset_test
