from __future__ import print_function

import numpy as np

from preprocessing import Preprocess
from models import model_dict

from sklearn.metrics import roc_auc_score

n = 14000
p = Preprocess()
train, test, output_cols, output_max = p.train, p.test, p.output_cols, p.output_max
TRAIN_SIZE = train.shape[0]


def try_model(model, n=14000, output_col=output_max, train=train):
    model.fit(train[:n], output_col[:n])
    if n < TRAIN_SIZE:
        output_pred = model.predict_proba(train[n:])[:, 1]
        print("roc auc on test: {}\n".format(roc_auc_score(output_col[n:],
                                             output_pred)))

    return model.predict_proba(test)[:, 1]


def constrained_perm(output_col, n, threshold=0.7):
    indices_one = np.where(output_col > 0)[0]
    indices_zero = np.where(output_col == 0)[0]
    np.random.shuffle(indices_zero)
    tt = int(threshold * indices_one.shape[0])
    perm = np.random.permutation(len(indices_one))
    kept = indices_one[perm[:tt]]
    discarded = indices_one[perm[tt:]]
    res = np.array(list(kept) + list(indices_zero) + list(discarded))
    return res


def shuffle_in_unison(a, b, n=14000, threshold=0.7):
    # copied from stack overflow
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = constrained_perm(b, n=n, threshold=threshold)
    for old_index, new_index in enumerate(permutation):
        shuffled_a[old_index] = a[new_index]
        shuffled_b[old_index] = b[new_index]
    return shuffled_a, shuffled_b


def batch_test(nb_iter=10, model_dict=model_dict):
    res = {}
    for name, m in zip(model_dict.keys(), model_dict.values()):
        res[name] = np.empty([3, nb_iter], dtype=float)
        for j in range(nb_iter):
            for i in range(3):
                tt, output_col = shuffle_in_unison(train.values, output_cols[i], n)
                m.fit(tt[:n], output_col[:n])
                output_pred = m.predict_proba(tt[n:])[:, 1]
                auc = roc_auc_score(output_col[n:], output_pred)
                print("roc auc on test: {}\n".format(auc))
                res[name][i][j] = auc
    return res


def summary(dic, path=False):
    s = ""
    for name in dic.keys():
        s += "SUMMARY OF: " + name + "\n"
        for i in range(3):
            s += "sum and variance for {}: {} -- {}\n".format(i, dic[name][i].mean(), dic[name][i].std())

    print(s)
    if path:
        log = open(path, "w")
        print(s, file=log)

