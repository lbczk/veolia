import numpy as np


class Meta:
    def __init__(self, model_list, agregator="sum"):
        self.model_list = model_list
        self.agregator = agregator
        assert agregator in ["sum", "max"]

    def fit(self, X, y):
        for m in self.model_list:
            m.fit(X, y)

    def predict_proba(self, X):
        res = np.zeros([X.shape[0], 2], dtype=np.float64)
        for m in self.model_list:
            if self.agregator == "sum":
                res += m.predict_proba(X)
            else:
                res = np.maximum(res, m.predict_proba(X))
        return res
