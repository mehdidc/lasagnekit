import os
from .imagecollection import ImageCollection


class FlatIcon(ImageCollection):
    folder = os.path.join(os.getenv("DATA_PATH"),
                          "flaticon")
    output_dim = 500

    def process_dirs(self, dirs):
        dirs = filter(lambda d: os.path.isdir(d), dirs)
        self.dir_to_label = {d: i for i, d in enumerate(dirs)}
        dirs = map(lambda d: d + "/png", dirs)
        return dirs

    def filename_to_label(self, directory, filename=""):
        directory_ = os.path.dirname(directory)
        return self.dir_to_label[directory_]
