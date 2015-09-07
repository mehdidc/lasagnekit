import os
import numpy as np
import pandas as pd
import urllib
import re

class UCI(object):

    def __init__(self, dataset_filter=None, verbose=0):
        # a function which takes a dict describing
        # an UCI dataset and returns True or False
        if dataset_filter is None:
            dataset_filter = lambda x: True
        self.dataset_filter = dataset_filter
        self.verbose = verbose

    def load(self):
        df = pd.read_csv(os.path.join(os.getenv("DATA_PATH"), "uci", "uci.csv"))
        df = df.to_dict(orient="records")
        datasets = filter(self.dataset_filter, df)
        self.datasets = []
        for dataset in datasets:
            name = dataset["Name"]
            slug_name = name.split("(")[0]
            slug_name = slug_name.strip()
            slug_name = "-".join(slug_name.split(" "))
            slug_name = slug_name.lower()

            filename = os.path.join(os.getenv("DATA_PATH"), "uci", slug_name + ".data")
            if os.path.exists(filename + ".err"):
                continue
            orig_filename = filename
            if not os.path.exists(filename):

                try:
                    url_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/{0}".format(slug_name)
                    url_index=  "{0}/Index".format(url_base)
                    url_data = "{0}/{1}.data".format(url_base, slug_name)
                    try:
                        print(url_data)
                        urllib.URLopener().retrieve(url_data, filename)
                    except Exception:
                        index = urllib.URLopener().open(url_index)
                        content = index.read()
                        index.close()
                        data_filename = re.search(r'[0-9]{4}\s*[0-9]*\s(.*?\.data)', content)
                        data_filename = data_filename.groups()[0]
                        url_data = "{0}/{1}".format(url_base, data_filename)
                        urllib.URLopener().retrieve(url_data, filename)
                except Exception:
                    filename = None
            if filename is None:
                open(orig_filename + ".err", "w").close()
                if self.verbose > 0:
                    print("skipping {0}..because it does not exist".format(dataset["Name"]))
                continue
            if self.verbose > 0:
                print("loading {0}...".format(dataset["Name"]))
            try:
                d = pd.read_csv(filename, header=None)
            except Exception:
                if self.verbose > 0:
                    print("could not load {0} it with pandas... skipping".format(dataset["Name"]))
                continue
            X, y = d.iloc[:, 1:], d.iloc[:, 0]
            X=np.array(X.values)
            y=np.array(y.values)
            if np.prod(X.shape) == 0 or np.prod(y.shape) == 0:
                print("skipping {0}, Xshape or yshape have zero dims".format(dataset["Name"]))
                continue
            if self.verbose > 0:
                print("ok for {0}".format(dataset["Name"]))
            self.datasets.append((X, y))

if __name__ == "__main__":
    uci = UCI(verbose=1)
    uci.load()
    print(len(uci.datasets))
