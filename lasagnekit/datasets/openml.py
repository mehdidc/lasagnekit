import os
import numpy as np
import pandas as pd
import urllib
import re
import json
import cPickle as pickle

def get_result_as_dict(url):
    u = urllib.urlopen(url)
    j = json.load(u)
    u.close()
    return j


class OpenML(object):

    def __init__(self, query="SELECT did FROM dataset", verbose=0):
        self.query = query
        self.verbose = verbose

    def load(self):
        ids_filename = os.path.join(os.getenv("DATA_PATH"), "openml", "ids.pkl")
        if not os.path.exists(ids_filename):
            url = "http://www.openml.org/api_query/?{0}".format(urllib.urlencode({"q": self.query}))
            result = get_result_as_dict(url)
            f = open(ids_filename, "w")
            pickle.dump(result, f)
        else:
            result = pickle.load(open(ids_filename, "r"))

        ds_ids = [int(r[0]) for r in result["data"]]

        for ds_id in ds_ids:
            if self.verbose:
                print("retrieving {0}...".format(ds_id))
            url_desc = "http://www.openml.org/d/{0}/json".format(ds_id)
            u = urllib.urlopen(url_desc)
            desc = json.load(u)
            u.close()
            if "arff" not in desc["url"]:
                if self.verbose:
                    print("skipping {0}...".format(ds_id))
                continue
            filename = os.path.join(os.getenv("DATA_PATH"), "openml", "{0}.arff".format(ds_id))
            if not os.path.exists(filename):
                urllib.urlretrieve(desc["url"], filename)

if __name__ == "__main__":
    openml = OpenML(verbose=1)
    openml.load()
