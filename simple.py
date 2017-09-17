import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time,math

from sklearn import model_selection , preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.mlab as mlab
import random

# MODELS
from sklearn.linear_model import LogisticRegressionCV as Logit
from sklearn.ensemble import GradientBoostingClassifier as GB

# DIMENSION REDUCTION
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
output = pd.read_csv("data/output_train.csv", delimiter=";")

TRAIN_SIZE = train.shape[0]

categorical_columns = [d for d in train.columns if train[d].dtype=='object']

total = train.append(test)

# year_construction = total["YearConstruction"].values
# year_construction =  np.array([str(u % 40) for u in year_construction], dtype=object)
# # total["year_const_vect"] = year_construction

total = categorical(total)
del total['Id']

# Dealing with the year of last failure
a = total["YearLastFailureObserved"].values
b = a.copy()
c = a.copy()
for i in range(a.shape[0]):
	b[i] = 0 if math.isnan(a[i]) else 1
	c[i] =  0 if math.isnan(a[i]) else 2018- a[i]

# exponential scaling of the length
def exp_scale(tab, maxx=15.):
	return np.array([np.exp(0.2+ min(u,maxx)/10.) for u in tab])
explength = exp_scale(total["Length"].values)
total["YearLastFailureObserved"] = b
total["foo"] = c
total["explength"] = explength


new_train, new_test = total[:TRAIN_SIZE], total[TRAIN_SIZE:]

# OUTPUT COLUMNS
# 2014 is output_cols[Ø] and 2015 is output_cols[1]
output_cols=[output.values[:,i] for i in [1,2]]
output_max=np.maximum(output_cols[0],output_cols[1])

def try_model(model, N=14000, output_col=output_max):
	model.fit(new_train[:N], output_col[:N])
	if N<TRAIN_SIZE:
		output_pred = model.predict_proba(new_train[N:])[:,1]
		print "roc auc on test: {}\n".format(roc_auc_score(output_col[N:], output_pred))

	return model.predict_proba(new_test)[:,1]

model_dict = {"logit":Logit(), "GB":GB(n_estimators=20)}

for u in model_dict.keys():
	print "Trying model: "+ u
	for i in [0,1]:
		try_model(model_dict[u], N=14000, output_col=output_cols[i])

res = try_model(model_dict["logit"], N=TRAIN_SIZE)
res_df = pd.DataFrame({"Id":test["Id"], "2014":res, "2015":res})
res_df.to_csv("predictions/"+"GB"+".csv",sep=";", columns=["Id", "2014", "2015"],index=False)

# aggregate_models()

res_logit = [try_model(model_dict["logit"], N=14*10**3, output_col=output_cols[i]) for i in [0,1]]
res_gb = [try_model(model_dict["GB"], N=14*10**3) for i in [0,1]]

res_df = pd.DataFrame({"Id":test["Id"], "2014":res[0], "2015":res[1]})
res_df.to_csv("predictions/last.csv",sep=";", columns=["Id", "2014", "2015"],index=False)

# NOT USED YET
def aggregate_models(model_list, coeff_vector, test):
	r = np.zeros(test.shape[0])
	for model in model_list:
		r += coeff_vector * model.predict(test)
	return r


def fit_models(model_list, train, output_col):
	for model in model_list:
		model.fit(train, output_col)

