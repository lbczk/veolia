import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import model_selection , preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.mlab as mlab
import random

# MODELS
from sklearn.linear_model import LogisticRegression as Logit

# DIMENSION REDUCTION
from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
output = pd.read_csv("data/output_train.csv", delimiter=";")

TRAIN_SIZE = train.shape[0]

categorical_columns = [d for d in train.columns if train[d].dtype=='object']

total = train.append(test)
total = categorical(total)
del total['Id']

# Need to deal with this later
del total["YearLastFailureObserved"]
new_train, new_test = total[:TRAIN_SIZE], total[TRAIN_SIZE:]

output_cols=[output.values[:,i] for i in [1,2]]
output_max=np.maximum(output_cols[0],output_cols[1])

model = Logit()
model.fit(train, output_max)

