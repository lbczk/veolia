# Implements a local search to optimize the AUC
import numpy as np

from sklearn.metrics import roc_auc_score


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


class Local:
    def __init__(self, train, test, output_col, init_weights):
        self.init_weights = init_weights.copy()
        self.weights = init_weights.copy()
        self.train = train
        self.test = test
        self.output_col = output_col
        self.multipliers = [0.9, 1.05]
        self.score = self.auc_score(self.init_weights)
        self.step = 1.2
        print("Initial AUC: {}\n".format(self.score))

        assert self.weights.shape[0] == train.shape[1]

    def auc_score(self, weights):
        return roc_auc_score(self.output_col, np.dot(self.train, weights.T))

    # def try_modifying(self, i, verbose=False):
    #     new_weights = self.weights.copy()
    #     for m in self.multipliers:
    #         # print "Dealing with coordinate {}. Trying to improve by multiplying by {}\n".format(i, m)
    #         new_weights[i] = m * self.weights[i]
    #         new_score = self.auc_score(new_weights)
    #         if new_score > self.score:
    #             if verbose:
    #                 print("Dealing with coordinate {}. Managed to improve by multiplying by {}\n".format(i, m))
    #             self.score = new_score
    #             self.weights = new_weights
    #             break

    def try_modifying_2(self, verbose=False):
        t = self.weights.shape[0]
        scores = np.empty(t, dtype=np.float64)
        new_weights = self.weights.copy()
        for i in range(self.weights.shape[0]):
            new_weights = self.weights.copy()
            new_weights[i] = self.step * self.weights[i]
            scores[i] = self.auc_score(new_weights)

        if scores.max() > self.score:
            self.score = scores.max()
            i = scores.argmax()
            self.weights[i] = self.step * self.weights[i]
            return 0
        else:
            u = np.random.rand()
            if u < 0.4:
                self.step = 1 + (self.step - 1) * 0.8
                return 1
            elif 0.4 < u < 0.6:
                self.step = 1 + 3 * (self.step - 1)
                return 2
            else:
                self.step = 2 - self.step
                return 3

    def search(self, nb_iter=10, verbose=False):
        n = 0
        n_last = 0
        while n < nb_iter:
            u = self.try_modifying_2(verbose)
            if verbose:
                print("ITERATION {} resulted in {}".format(n, u))
            if u == 0:
                n_last = n
            n += 1

        print("FINISHED\n-- final auc on train: {}\n".format(self.auc_score(self.weights)))
        print("n_last = {}  -- step = {} ".format(n_last, self.step))
        return np.dot(self.test, self.weights.T)


###################
# N = 14000
# logit = Logit()
# logit.fit(new_train[:N], output_max[:N])

# weights = logit.coef_[0]
# loc = Local(weights,1, new_train[:N], new_train[N:],output_max[:N])
# pred = loc.search(nb_iter=1000, verbose=False)

# print "Previously on TEST, AUC was {}".format(roc_auc_score(output_max[N:], np.dot(new_train[N:], weights.T)))

# print "New AUC on TEST is {}".format(roc_auc_score(output_max[N:], pred))

# loc = Local(res, res, output_col[N:], init_weights=np.array([1, 1, 5, 1, 1]))
# pred = lcl.search(nb_iter=400, verbose=False)

# print "Previously on TEST, AUC was {}".format(roc_auc_score(oc[n:], np.dot(train_intern, weights.T)))
# print "New AUC on TEST is {}".format(roc_auc_score(oc[n:], pred))
