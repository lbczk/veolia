# MODELS

from sklearn.ensemble import AdaBoostClassifier as Ada
from sklearn.ensemble import ExtraTreesClassifier as Extra
from sklearn.ensemble import GradientBoostingClassifier as GradB
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.linear_model import LogisticRegression as Logit

from sklearn import svm

from meta import Meta

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
# Dictionaries of models
####

small_model_dict = {"logit": Logit()}

model_dict = {"logit": Logit(tol=10**(-6)),
              "GradB": GradB(**GradB_params),
              "forest": Forest(**forest_params),
              "forest_2": Forest(**forest_params),
              "ada": Ada(**ada_params),
              "extra": Extra(**extra_params),
              "extra_entropy": Extra(criterion="entropy", **extra_params),
              "extra_3": Extra(min_samples_split=5, **extra_params),
              "extra_4": Extra(criterion="entropy", min_samples_split=5, n_estimators=40),
              }

extended_dict = {"logit_": Logit(),
                 "extra_entropy_": Extra(criterion="entropy", min_samples_split=15, max_depth=4, n_estimators=100),
                 "extra_40_entropy": Extra(criterion="entropy", min_samples_split=10, max_depth=5, n_estimators=20),
                 "extra_40_entropy": Extra(criterion="entropy", min_samples_split=10, max_depth=3, n_estimators=30)
                 }

master_dict = dict(model_dict.items() + extended_dict.items())

meta_dict = {
    "logit": Logit(tol=10**(-6)),
    "a": Meta([Logit(tol=10**(-6)), Extra(**extra_params), Logit(tol=10**(-6))]),
    "b": Meta([Logit(tol=10**(-6)), Forest(**forest_params)])
}

meta_dict_2 = {
    "a_max": Meta(agregator="max", model_list=[Logit(tol=10**(-6)), Extra(**extra_params), Forest(**forest_params), Extra(criterion="entropy", min_samples_split=5, n_estimators=40)]),
    "b_max": Meta(agregator="max", model_list=[Logit(tol=10**(-6)), Forest(**forest_params)])
}

master_meta_dict = dict(meta_dict.items() + meta_dict_2.items())

svm_dict = {
    "logit": Logit(tol=10**(-6)),
    "svc_1": svm.SVC(C=0.2, probability=True, kernel="linear"),
    "svc_2": svm.SVC(C=0.4, probability=True, kernel="linear"),
    "svc_3": svm.SVC(C=1., probability=True, kernel="linear"),
    "svc_4": svm.SVC(C=2., probability=True, kernel="linear")
}
