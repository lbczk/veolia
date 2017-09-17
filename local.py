# Implements a local search to optimize the AUC
import pandas as pd
import numpy as np
import math

from sklearn.metrics import roc_auc_score

def sigmoid_array(x):
	return 1/(1+np.exp(-x))

class Local:
	def __init__(self, init_weights, intercept, train, test, output_col):
		self.init_weights = init_weights.copy()
		self.weights = init_weights.copy()
		self.train = train
		self.test = test
		self.output_col = output_col
		self.intercept = intercept

		self.multipliers = [ 0.95, 1.05, 0.99, 1.001]

		self.score =  self.auc_score(self.init_weights)

		print "Initial AUC: {}\n".format(self.score)

	def auc_score(self, weights):
		"""
		"""
		# probas = sigmoid_array(np.dot(self.train, weights.T) + self.intercept)
		return roc_auc_score(self.output_col, np.dot(self.train, weights.T))
		

	def try_modifying(self, i, verbose=False):
		new_weights = self.weights.copy()
		for m in self.multipliers:
			# print "Dealing with coordinate {}. Trying to improve by multiplying by {}\n".format(i, m)
			new_weights[i] = m* self.weights[i]
			new_score = self.auc_score(new_weights)
			if new_score > self.score:
				if verbose:
					print "Dealing with coordinate {}. Managed to improve by multiplying by {}\n".format(i, m)
				self.score = new_score
				self.weights = new_weights
				break

	def search(self, nb_iter=10, verbose=False):
		n=0
		i=5
		while n < nb_iter:
			if verbose:
				print "STARTING ITERATION {}".format(n)
			self.try_modifying(i, verbose)
			i = (i + 1)% self.weights.shape[0]
			n += 1

		print "FINISHED\n -- final auc on train: {}".format(self.auc_score(self.weights))

		return np.dot(self.test, self.weights.T)

N = 14000
loc = Local(weights,1, new_train[:N], new_train[N:],output_max[:N])
pred = loc.search(nb_iter=400, verbose=False)

print roc_auc_score(output_max[N:], pred)
print roc_auc_score(output_max[N:], np.dot(new_train[N:], weights.T))
