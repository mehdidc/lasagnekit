import os, sys, tarfile, urllib
import numpy as np
import matplotlib.pyplot as plt
# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = "{}/stl10".format(os.getenv("DATA_PATH"))

# path to the binary train file with image data
DATA_PATH = DATA_DIR + '/stl10_binary/train_X.bin'
DATA_TEST_PATH = DATA_DIR + '/stl10_binary/test_X.bin'


DATA_UNLABELED_PATH = DATA_DIR + '/stl10_binary/unlabeled_X.bin'

# path to the binary train file with labels
LABEL_PATH = DATA_DIR + '/stl10_binary/train_y.bin'
LABEL_TEST_PATH = DATA_DIR + '/stl10_binary/test_y.bin'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


class STL(object):
    
    def __init__(self, kind="train_labeled"):
        assert kind in ("train_labeled", "test_labeled", "unlabeled")
        self.kind = kind

    def load(self):
        if self.kind.endswith("_labeled"):
            images = read_all_images(DATA_PATH if self.kind.startswith("train") else DATA_TEST_PATH)
            labels = read_labels(LABEL_PATH if self.kind.startswith("train") else LABEL_TEST_PATH)
            self.output_dim = 10
            self.y = labels
        else:
            images = read_all_images(DATA_UNLABELED_PATH)
        images = images.astype(np.float32)
        self.img_dim = images.shape[1:]
        self.X = images.reshape((images.shape[0], -1))
