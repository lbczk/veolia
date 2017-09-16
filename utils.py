import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# MODELS
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression as LR

def cat_vectorize(values):
	"""
	Transforms an array of categorical values into an array of indicator vectors
	"""
	values = np.array(values)
	maximum = values.max()
	res = np.zeros([values.shape[0], maximum])
	for i in range(values.shape[0]):
		res[i][values[i] - 1] = 1
	return res

# replacing categorical by a value between 0 and n_features - 1
def categorical_naive(dataframe):
	dataframe = dataframe.copy()
	for f in dataframe.columns:
	    if dataframe[f].dtype == 'object':
	        lbl = preprocessing.LabelEncoder()
	        lbl.fit(list(dataframe[f].values))
	        dataframe[f+"_naive"] = lbl.transform(list(dataframe[f].values))
	return dataframe

def categorical(dataframe, threshold=50):
	"""
	Replaces the categorical columns in dataframe by vectorized version. 
	Removes values appearing less than threshold
	"""
	dataframe = dataframe.copy()
	for f in dataframe.columns:
	    if dataframe[f].dtype == 'object':
	    	lbl = preprocessing.LabelEncoder()
	    	values = dataframe[f].values
	    	values = filter_values(values, threshold, default="default")
	        lbl.fit(list(values))
	        values = lbl.transform(list(values))
	        # filter values below threshold
	        res = cat_vectorize(values)
	        print "DOING IT FOR " , f , "--> " , res.shape[0] , res.shape[1]
	        for i in range(res.shape[1]):
	        	dataframe[f+"_"+str(i)] = res[:,i]
	    if f != "y":
	    	del dataframe[f]   
	return dataframe


def noncategorical(dataframe):
	dataframe = dataframe.copy()
	for f in dataframe.columns:
	    if dataframe[f].dtype == 'object' or f == "y":
	    	del dataframe[f]   
	return dataframe

from collections import Counter

def filter_values(values, threshold=50, default="DEF"):
	"""
	Remove values below a given threshold
	"""
	values = np.array(values)
	count = Counter(values)
	res =  values.copy()
	for i in range(values.shape[0]):
		if count[values[i]] < threshold:
			res[i] = default
	return res
