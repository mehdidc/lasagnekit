import os
import numpy as np

class Cifar10(object):
    BATCH_INDEX_MAPPING = {
            1: "data_batch_1",
            2: "data_batch_2",
            3: "data_batch_3",
            4: "data_batch_4",
            5: "data_batch_5",
            6: "test_batch",
    }
    NB_EXAMPLES_PER_BATCH = 10000

    def __init__(self, batch_indexes=None, train_or_test=None):
        # either batch_indexes or train_or_test must have a value, not both in the same time
        assert ( (batch_indexes is None and train_or_test is not None) or
                 (batch_indexes is not None and train_or_test is None) )

        if batch_indexes is not None:
            pass
        else:
            if train_or_test == 'train':
                batch_index = range(1, 6)
            elif train_or_test == 'test':
                batch_indexes = [6]

        self.data_filenames = map(lambda i:Cifar10.BATCH_INDEX_MAPPING[i], batch_indexes)

    def load(self):

        nb_batches = len(self.data_filenames)
        nb_examples = nb_batches * Cifar10.NB_EXAMPLES_PER_BATCH
        X = np.zeros( (nb_examples, 3072)  )
        y = np.zeros( (nb_examples,) )

        start = 0
        end = Cifar10.NB_EXAMPLES_PER_BATCH
        for data_filename in self.data_filenames:
            file_content = np.load(os.path.join(os.getenv("DATA_PATH"), "cifar10", data_filename))
            X[start:end] = file_content.get("data")
            y[start:end] = file_content.get("labels")
            start = end
            end += Cifar10.NB_EXAMPLES_PER_BATCH
        self.img_dim = (3, 32, 32)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
