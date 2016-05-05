import os
from .imagecollection import ImageCollection
import re

dir_regex = re.compile('character[0-9]{2}$')


class Omniglot(ImageCollection):
    folder = os.path.join(os.getenv("DATA_PATH"),
                          "omniglot",
                          "images_background")
    output_dim = 52

    def __init__(self, *args, **kwargs):
        kwargs['recur'] = True
        #kwargs["size"] = None
        super(Omniglot, self).__init__(*args, **kwargs)

    def process_dirs(self, dirs):
        dirs = filter(lambda d: os.path.isdir(d) and dir_regex.search(d), dirs)
        return dirs

    def filename_to_label(self, directory, filename=""):
        return directory
