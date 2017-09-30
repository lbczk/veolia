import numpy as np

from collections import Counter

from sklearn import preprocessing

# def cat_vectorize(values):
# 	"""
# 	Transforms an array of categorical values into an array of indicator vectors.
# 	"""
# 	values = np.array(values)
# 	maximum = values.max()
# 	res = np.zeros([values.shape[0], maximum + 1])
# 	for i in range(values.shape[0]):
# 		res[i][values[i]] = 1
# 	return res


def cat_vectorize(values):
	"""
	Transforms an array of categorical values into an array of indicator vectors.

	INPUT:
	- values array-object of categorical
	"""
	values = np.array(values).reshape(-1, 1)
	enc = preprocessing.OneHotEncoder()
	enc.fit(values.reshape(-1, 1))
	return enc.transform(values).toarray()


def filter_values(values, threshold=50, default="DEF"):
	"""
	filter_values returns an array where only values appearing more than threshold times
	are kept. Other entries are replaced by a value called default

	INPUT:
	- values, an array containing values of an arbitrary type
	- threshold, an int
	- default, the default value
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

	INPUT:
	- dataframe, a pandas object
	- threshold, an int
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
			print("OneHotEncoding {} --> {} rows, {} cols".format(f, res.shape[0], res.shape[1]))
			for i in range(res.shape[1]):
				dataframe[f + "_" + str(i)] = res[:, i]
			if f != "y":
				del dataframe[f]
	return dataframe
