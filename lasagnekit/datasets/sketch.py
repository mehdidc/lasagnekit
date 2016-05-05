import os
from .imagecollection import ImageCollection


class Sketch(ImageCollection):
    folder = os.path.join(os.getenv("DATA_PATH"),
                          "sketch")
    output_dim = 252

    def process_dirs(self, dirs):
        dirs = filter(lambda d: os.path.isdir(d), dirs)
        self.dir_to_label = {d: i for i, d in enumerate(dirs)}
        return dirs

    def filename_to_label(self, directory, filename=""):
        return self.dir_to_label[directory]
