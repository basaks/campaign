import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import config


def select_features_rfe(model, X, y):
    """
    Select features using RFE for a particular, and data set X and y
    :param model: sklearn.classfication instance
        classification model used for feature selection
    :param X: pd.DataFrame object
        Data Matrix
    :param y:
    :return:
    """
    # check X is data frame or raise/exception handling
    rfe_balanced_clf = RFE(model, config.no_selected_rfe)
    rfe = rfe_balanced_clf.fit(X, y)
    return [c for c, s in zip(X.columns, rfe.support_) if s]


def standardise_features(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def logit_filter(X, y):

    alpha = config.alpha  # alpha significance level

    exog = sm.add_constant(X)  # add intercept
    logit_model = sm.Logit(y, exog=exog)
    result = logit_model.fit_regularized(maxiter=1000, method='l1')
    # print(result.summary2())
    significant_cols = list(result.params.index[result.pvalues < alpha])
    if 'const' in significant_cols:
        significant_cols.remove('const')  # discard intercept
    return significant_cols
