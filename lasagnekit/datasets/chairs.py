import os
from .imagecollection import ImageCollection


class Chairs(ImageCollection):
    folder = os.path.join("chairs",
                          "rendered_chairs")
    output_dim = 1393

    def process_dirs(self, dirs):
        dirs = filter(lambda d: os.path.isdir(d), dirs)
        self.dir_to_label = {d: i for i, d in enumerate(dirs)}
        dirs = map(lambda d: d + "/renders", dirs)
        return dirs

    def filename_to_label(self, directory, filename=""):
        directory_ = os.path.dirname(directory)
        return self.dir_to_label[directory_]
