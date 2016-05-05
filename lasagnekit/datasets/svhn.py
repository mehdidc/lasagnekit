import os
from .imagecollection import ImageCollection


class SVHN(ImageCollection):

    folder = os.path.join(os.getenv("DATA_PATH"), "svhn")

    def __init__(self, which='train', *args, **kwargs):
        self.folder = os.path.join(self.folder, which)
        super(SVHN, self).__init__(*args, **kwargs)

    def filename_to_label(self, directory, filename):
        return 1
