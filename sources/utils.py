import numpy as np

from collections import Counter

from sklearn import preprocessing


def cat_vectorize(values):
	"""
	Transforms an array of categorical values into an array of indicator vectors
	"""
	values = np.array(values)
	maximum = values.max()
	res = np.zeros([values.shape[0], maximum + 1])
	for i in range(values.shape[0]):
		res[i][values[i]] = 1
	return res


def filter_values(values, threshold=50, default="DEF"):
	"""
	values is an array containing values of an arbitrary type
	filter_values returns an array where only values appearing more than threshold times
	are kept. Other entries are replaced by a value called default
	"""
	values = np.array(values)
	count = Counter(values)
	res = values.copy()
	for i in range(values.shape[0]):
		if count[values[i]] < threshold:
			res[i] = default
	return res


def categorical(dataframe, threshold=-1):
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
	        print("DOING IT FOR ", f, "--> ", res.shape[0], res.shape[1])
	        for i in range(res.shape[1]):
	        	dataframe[f + "_" + str(i)] = res[:, i]
	    	if f != "y":
	    		del dataframe[f]
	return dataframe