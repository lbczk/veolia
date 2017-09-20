# Read and preprocess the data
# The main preprocessing operation is dealing with categorical variables

from __future__ import print_function

import math

import numpy as np

import pandas as pd

from utils import categorical


# used to scale the length
def exp_scale(tab, maxx=100.):
    return np.array([np.exp(0.2 + min(u, maxx) / 10.) for u in tab])


class Preprocess:
    def __init__(self):

        self.raw_train = pd.read_csv("../data/train.csv")
        self.raw_test = pd.read_csv("../data/test.csv")
        self.raw_output = pd.read_csv("../data/output_train.csv", delimiter=";")
        TRAIN_SIZE = self.raw_train.shape[0]
        total = self.raw_train.append(self.raw_test)

        total = categorical(total)
        del total['Id']

        # Dealing with the year of last failure
        a = total["YearLastFailureObserved"].values
        b = a.copy()
        c = a.copy()
        for i in range(a.shape[0]):
            b[i] = 0 if math.isnan(a[i]) else 1
            c[i] = 0 if math.isnan(a[i]) else 2018 - a[i]

        explength = exp_scale(total["Length"].values)
        total["YearLastFailureObserved"] = b
        total["foo"] = c
        total["explength"] = explength

        self.train, self.test = total[:TRAIN_SIZE], total[TRAIN_SIZE:]

        self.output_cols = [self.raw_output.values[:, i] for i in [1, 2]]
        self.output_max = np.maximum(self.output_cols[0], self.output_cols[1])
        self.output_cols += [self.output_max]
