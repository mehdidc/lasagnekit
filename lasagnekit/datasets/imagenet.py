import os
import pandas as pd
from .imagecollection import ImageCollection

synonyms = pd.read_table(os.path.join(os.getenv("DATA_PATH"),
                         "imagenet", "words.txt"), header=None,
                         index_col=0)


class ImageNet(ImageCollection):
    folder = os.path.join("imagenet",
                          "imagenet_downloader")

    def process_dirs(self, dirs):
        return filter(lambda d: os.path.basename(d).startswith("n"), dirs)

    def filename_to_label(self, directory, filename):
        s = synonyms[1][os.path.basename(directory)]
        s = s.split(",")
        return s
