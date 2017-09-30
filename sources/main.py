from __future__ import print_function

import numpy as np

from preprocessing import Preprocess
from models import small_model_dict

from sklearn.metrics import roc_auc_score

n = 14000  # number of lines we use for an internal train.
p = Preprocess(normalize_params={"norm_list": ["l1"], "axis": [0], "inplace": False})
train, test, output_cols, output_max = p.train, p.test, p.output_cols, p.output_max
TRAIN_SIZE = train.shape[0]


def fit_and_predict(model, n=n, output_col=output_max, train=train):
    """
    Fit and predict model on the train data split at index n.
    NOT USED in the current script version.

    INPUT:
    - model is a sklearn model
    - n, an int
    - output_col, an array like object. The predicted column
    - train data (either pd.DataFrame or np.array)
    """
    model.fit(train[:n], output_col[:n])
    if n < TRAIN_SIZE:
        output_pred = model.predict_proba(train[n:])[:, 1]
        print("roc auc on test: {}\n".format(roc_auc_score(output_col[n:],
                                             output_pred)))

    return model.predict_proba(test)[:, 1]


def constrained_perm(output_col, n, threshold=0.7):
    """
    Permute the (binary) output column keeping a fixed proportion of ones and zeros.
    """
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
    """
    Given two array like objects a and b of dimension (x, _)
    shuffle them using the same constrained permutation on b.

    - This is essentially copied from stack overflow
    """
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = constrained_perm(b, n=n, threshold=threshold)
    for old_index, new_index in enumerate(permutation):
        shuffled_a[old_index] = a[new_index]
        shuffled_b[old_index] = b[new_index]
    return shuffled_a, shuffled_b


def batch_test(nb_iter=10, model_dict=small_model_dict, indices=[2]):
    """
    Run nb_iter tests using all models in model_dict.
    (minor) The indices parameter is used to specify which output columns
    to use in the training.
    """
    res = {}
    for name, m in zip(model_dict.keys(), model_dict.values()):
        res[name] = np.empty([len(indices), nb_iter], dtype=float)
        for j in range(nb_iter):
            for i in range(len(indices)):
                tt, output_col = shuffle_in_unison(train.values, output_cols[i], n)
                m.fit(tt[:n], output_col[:n])
                output_pred = m.predict_proba(tt[n:])[:, 1]
                auc = roc_auc_score(output_col[n:], output_pred)
                print("roc auc on test: {}\n".format(auc))
                res[name][i][j] = auc
    return res


def summary(dic, path=False, indices=[2]):
    """
    Print a nice summary of results contained in a dictionary dic.

    INPUT:
    - dic is a dictionary of results as output by batch test
    - path indicates where to write the result
    """
    s = ""
    for name in dic.keys():
        s += "SUMMARY OF: " + name + "\n"
        for i in range(len(indices)):
            s += "sum and variance for {}: {} -- {}\n".format(i, dic[name][i].mean(), dic[name][i].std())

    print(s)
    if path:
        log = open(path, "w")
        print(s, file=log)

if __name__ == '__main__':
    # replace small_model_dict by any other dict from models
    dic_results = batch_test(nb_iter=5, model_dict=small_model_dict, indices=[2])
    summary(dic_results, indices=range(1))

    # logit = model_dict["logit"]
    # pred_logit_2014 = fit_and_predict(logit, n=TRAIN_SIZE, output_col=output_cols[0])
    # pred_logit_2015 = fit_and_predict(logit, n=TRAIN_SIZE, output_col=output_cols[2])
    # res_logit = pd.DataFrame({"Id": range(19428, test.shape[0] + 19428), "2014": pred_logit_2014, "2015": pred_logit_2015})
    # res_logit.to_csv("../predictions/logit_norm.csv", sep=";", columns=["Id", "2014", "2015"], index=False)
