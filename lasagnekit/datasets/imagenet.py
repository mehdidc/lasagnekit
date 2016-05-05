import os
import pandas as pd
from .imagecollection import ImageCollection

synonyms = pd.read_table(os.path.join(os.getenv("DATA_PATH"),
                         "imagenet", "words.txt"), header=None,
                         index_col=0)


class ImageNet(ImageCollection):
    folder = os.path.join(os.getenv("DATA_PATH"), "imagenet", "imagenet_downloader")

    def __init__(self, *args, **kwargs):
        if "categories" in kwargs:
            categories = kwargs["categories"]
            self.categories = set(categories)
        else:
            categories = None
            self.categories = None
        super(ImageNet, self).__init__(*args, **kwargs)

    def process_dirs(self, dirs):
        if self.categories is None:
            return filter(lambda d: os.path.basename(d).startswith("n"), dirs)
        else:
            return filter(lambda d: os.path.basename(d) in self.categories, dirs)

    def filename_to_label(self, directory, filename):
        s = synonyms[1][os.path.basename(directory)]
        s = s.split(",")
        return os.path.basename(directory), s
