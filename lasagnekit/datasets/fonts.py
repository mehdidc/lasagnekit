import os
import numpy as np
import re
import lasagnekit

class Fonts(object):

    def __init__(self, kind='all_64', accept_only=None, labels_kind=None, start=None, stop=None):
        self.X = None
        self.y = None

        self.kind = kind
        self.accept_only = accept_only
        self.labels_kind = labels_kind
        self.start = start
        self.stop = stop

    def load(self):
        base_path = os.path.join(os.getenv("DATA_PATH"), "fonts")

        set_file ="ds_%s.npy" % (self.kind,)

        full_path = os.path.join(base_path, set_file)

        data = np.load(full_path)

        if len(data) == 2:
            data, labels = data
        else:
            labels = None

        if self.start is not None and self.stop is not None:
            data = data[self.start:self.stop]
            if labels is not None: labels = labels[self.start:self.stop]



        if labels is not None and self.accept_only is not None:
            mask = np.zeros(len(data)).astype(np.bool)
            for i, label in enumerate(labels):
                if re.match(self.accept_only, label):
                    mask[i] = True
                else:
                    mask[i] = False
        else:
            mask = None

        data = np.array(list(data))

        if labels is not None: labels = np.array(labels)
        if mask is not None:
            data = data[mask]
            if labels is not None:labels = labels[mask]


        if len(data.shape) == 2:
            w = int(np.sqrt(data.shape[1]))
            data = data.reshape((data.shape[0], w, w))
        self.img_dim = data.shape[1:]

        N = np.prod( data.shape[1:] )
        data = data.reshape(  (data.shape[0], N))

        orig_labels = labels
        if self.labels_kind == "letters":
            new_labels = []
            for label in labels:
                character_match = re.match('.*-([a-z])-.*', label)
                if character_match:
                    c = character_match.group(1)
                    c_id = ord(c) - ord('a')
                    new_labels.append(c_id)
                else:
                    print('warning : unkown label...')
                    new_labels.append(0)
            labels = np.array(new_labels)
            labels = labels.astype(np.int32)
            y_labels = 26
        else:
            y_labels = None

        self.X = 1 - data.astype(np.float32) / 255.
        self.X = lasagnekit.easy.linearize(self.X)
        self.y = labels
        self.y_raw = orig_labels
        self.output_dim = y_labels
