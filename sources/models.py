# MODELS

from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.ensemble import ExtraTreesClassifier as Extra
from sklearn.ensemble import GradientBoostingClassifier as GradB
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.linear_model import LogisticRegressionCV as Logit

#####
# Different parameters for different predictors
#####

GradB_params = {
    "n_estimators": 40
}

forest_params = {
    "n_estimators": 40,
    "max_depth": 4
}

extra_params = {
    "n_estimators": 60,
    "max_depth": 3
}

ada_params = {
    "n_estimators": 40
}

####
# All models we use, grouped into dictionaries
####
small_model_dict = {"logit": Logit()}

model_dict = {"logit": Logit(),
              "GradB": GradB(**GradB_params),
              "forest": Forest(**forest_params),
              "forest_2": Forest(**forest_params),
              "ada": Ada(**ada_params),
              "extra": Extra(**extra_params),
              "extra_entropy": Extra(criterion="entropy", **extra_params),
              "extra_3": Extra(min_samples_split=5, **extra_params),
              "extra_4": Extra(min_samples_split=10, n_estimators=40),
              }

extended_dict = {"logit_": Logit(),
                 "extra_": Extra(**extra_params),
                 "extra_entropy_": Extra(criterion="entropy", **extra_params),
                 "extra_3_entropy": Extra(criterion="entropy", min_samples_split=5, **extra_params),
                 "extra_80_entropy": Extra(criterion="entropy", min_samples_split=5, max_depth=3, n_estimators=80),
                 "extra_80_entropy": Extra(criterion="entropy", min_samples_split=10, max_depth=3, n_estimators=80)
                 }
