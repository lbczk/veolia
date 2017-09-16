import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import model_selection , preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
import matplotlib.mlab as mlab
import random

# MODELS
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as Forest
from sklearn.ensemble import ExtraTreesRegressor as XTrees
from sklearn.ensemble import GradientBoostingRegressor as GradBoost

# DIMENSION REDUCTION
from sklearn.decomposition import PCA

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
output = pd.read_csv("data/output_train.csv")