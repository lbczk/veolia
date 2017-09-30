from __future__ import print_function

import math

import numpy as np

import pandas as pd

from utils import categorical

from sklearn.preprocessing import normalize


class Preprocess:
    """
    Preprocessing the data contained at ../data/
    Train and test are stored as attributes of the object
    after the method compute_train_test is executed
    """
    def __init__(self, normalize_params={}):
        self.raw_train = pd.read_csv("../data/train.csv")
        self.raw_test = pd.read_csv("../data/test.csv")
        self.raw_output = pd.read_csv("../data/output_train.csv", delimiter=";")
        self.TRAIN_SIZE = self.raw_train.shape[0]
        total = self.raw_train.append(self.raw_test)

        del total["Feature4"]
        total = categorical(total)
        del total['Id']

        # Dealing with the year of last failure
        a = total["YearLastFailureObserved"].values
        b = a.copy()
        for i in range(a.shape[0]):
            b[i] = 0 if math.isnan(a[i]) else a[i]

        total["YearLastFailureObserved"] = b

        self.output_cols = [self.raw_output.values[:, i] for i in [1, 2]]
        self.output_max = np.maximum(self.output_cols[0], self.output_cols[1])
        self.output_cols += [self.output_max]
        self.total = total

        if normalize_params:
            self.normalize(**normalize_params)

        self.compute_train_test()

    def normalize(self, norm_list=["l1"], axis=[1], inplace=True):
        cols = self.total.columns
        total_copy = self.total.copy()
        for i, norm in enumerate(norm_list):
            total_copy = normalize(total_copy, norm=norm, axis=axis[i])
            addendum = "_normed_" + str(i) if not inplace else ""
            for j, u in enumerate(cols):
                self.total[u + addendum] = total_copy[:, j]

    def compute_train_test(self):
        self.train, self.test = self.total[:self.TRAIN_SIZE], self.total[self.TRAIN_SIZE:]
