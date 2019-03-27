from functools import partial, update_wrapper
from pathlib import Path
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, \
    SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import (make_scorer,
                             precision_score,
                             recall_score,
                             accuracy_score,
                             f1_score,
                             fbeta_score)
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier)
from xgboost import XGBClassifier
# keep adding more classifiers


from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, \
    ADASYN, RandomOverSampler
from imblearn.under_sampling import (RandomUnderSampler,
                                     TomekLinks,
                                     AllKNN,
                                     NearMiss,
                                     RepeatedEditedNearestNeighbours,
                                     EditedNearestNeighbours)

# Dataset balancing parameters

supported_samplers = {
    # oversamplers
    'smote': SMOTE(),
    'bsmote': BorderlineSMOTE(),
    'svmsmote': SVMSMOTE(),
    'adasyn': ADASYN(),
    'randomoversampler': RandomOverSampler(),
    # undersamplers
    'randomundersampler': RandomUnderSampler(),
    'tomek': TomekLinks(),
    'allknn': AllKNN(),
    'nearmiss1': NearMiss(version=1),
    'nearmiss2': NearMiss(version=2),
    'nearmiss3': NearMiss(version=3),
    'enn': EditedNearestNeighbours(),
    'repeatedenn': RepeatedEditedNearestNeighbours()
}


# instead of these samplers, could try just overweighting class 1
sampling_algo = supported_samplers['smote']


# list all supported classifiers
supported_classifiers = {
    'svc': SVC,
    'logit': LogisticRegression,
    'ridge': RidgeClassifier,
    'rf': RandomForestClassifier,
    'ada': AdaBoostClassifier,
    'gradientboost': GradientBoostingClassifier,
    'extratree': ExtraTreesClassifier,
    'xgboost': XGBClassifier
}


# choose classifier and properties

classifier_params = {
    # logistic regression
    'logit': {
        'solver': 'lbfgs',
        'fit_intercept': True,
        'multi_class': 'ovr',
        'max_iter': 50,
        'verbose': False,
        'n_jobs': 4,
        'tol': 1e-3,
        'C': 0.01,
        # approximate ratio of class 0/class 1
        # 'class_weight': {0: 1, 1: 7.9}
        },

    # random forest classifier
    'rf': {
        'n_estimators': 100,
        'n_jobs': -1,
        'min_samples_leaf': 5
        },

    # svc params
    'svc': {
        'gamma': 0.001,
        },

    # xgboost params
    'xgboost': {
        'n_estimators': 100,
        'n_jobs': -1,
        'min_samples_leaf': 3,
        'booster': 'gbtree'
        }
}


clf = 'logit'

classifier = supported_classifiers[clf](**classifier_params[clf])

# crossval parameters
optimise = True
crossval_folds = RepeatedKFold(n_splits=2, n_repeats=2)
# classifier optimiser grid when optimising
# lbfgs does not support l1, use l2

# this is for LogisticRegression
p_grid = {"penalty": ['l2'],
          "C": np.power(10, range(7))/10000}


# random forest grid
# p_grid = {
#     'min_samples_split': [3, 5, 10],
#     'n_estimators': [10, 30],
#     'max_depth': [3, 5],
#     'max_features': [3, 5]
# }


# p_grid = {
#     'min_samples_split': [3, 5, 10],
#     'n_estimators': [100, 200],
#     'max_depth': [3, 5, 15, 25],
#     'max_features': [3, 5, 10, 20]
# }


# feature selection attributes
# selected features for Random Projection/PCA/RFE
no_selected_rfe = 20


feature_selection = True
significance_test = False
# alpha is significance level of coefficient 0.1=10% during RFE
alpha = 0.1
random_projection = False


def wrapped_partial(func, *args, **kwargs):
    """
    custom functools.partial wrapper.

    We need a __name__ attribute which simple partial's don't have. Without a
    __name__ attribute in the score function, gridsearch does not work.

    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


fbeta2 = wrapped_partial(fbeta_score, beta=2)


# probability threshhold
scorers = {
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'accuracy': make_scorer(accuracy_score),
    'f1-score': make_scorer(f1_score),
    # fbeta2 is fscore with 2X weighted towards recall than  precision
    'fbeta2': make_scorer(fbeta2)
}

grid_search_criteria = 'f1-score'

# campaign_budget


# flag for doing analysis with and without campaign
only_campaign = True   # Set True to analyse only campaign (Group B customers)
only_non_campaign = False  # Set True to analyse (Group A - Group B) customers
# If both are set to False, all data (Group A) is selected with campaing
# boolean column

if only_campaign and only_non_campaign:
    raise AttributeError('Only one of only_campaign or only_non_'
                         'campaign can be true')

data_path = Path('campaign.csv')

all_available_columns = ["age", "job", "marital", "education", "default",
                         "housing", "loan", "campaign", "contact", "month",
                         "day_of_week", "duration", "previous", "poutcome",
                         "cons.price.idx", "cons.conf.idx", "y"]

# duration is removed from all models

selected_cols = ["age", "job", "marital", "education", "default", "housing",
                 "loan", "campaign", "contact",
                 'month', 'day_of_week', "previous", "poutcome",
                 "cons.price.idx", "cons.conf.idx", "y"]

all_categorical_features = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'day_of_week', 'poutcome']

selected_categorical_features = ['job', 'marital', 'education', 'default',
                                 'housing', 'loan', 'contact',
                                 'month', 'day_of_week', 'poutcome']

# discard `duration`
campaign_related_features = ["contact", "month", "day_of_week"]

selected_ordinal_features = ['age', "cons.price.idx", "cons.conf.idx", "y"]

int_features = ['campaign', 'previous']

# whether to discard one column from the one hot categories, this is known to
# improve the condition number of the data matrix
discard_dependent_category = False


# this is the ratio of FP/FN
optimum_multiplier = 10
