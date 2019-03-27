from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, train_test_split

import config
from features import select_features_rfe, logit_filter, standardise_features
import utils


def data_prep(conf):
    """
    :param conf: config module
    :return: X, y based on config
    """

    def _check_columns_after_dataprep():
        """
        some basic checks that we did not lose some columns in the
        tranformations
        """
        assert set(X_ordinal.columns).issubset(set(X_final.columns))
        assert set(X_ints.columns).issubset(set(X_final.columns))
        total_one_hot_categories = 0
        for c in X_cat.columns:
            # -1 due to dropped column for independent one hot columns
            total_one_hot_categories += len(pd.unique(X_cat[c])) - \
                                        int(conf.discard_dependent_category)
        assert len(X_final.columns) == total_one_hot_categories + len(
            X_ordinal.columns) + len(X_ints.columns)

    selected_cols = set(conf.selected_categorical_features +
                       conf.selected_ordinal_features)
    selected_cols.remove('y')
    ordinal_features = list(selected_cols
                            - set(conf.selected_categorical_features)
                            - set(conf.int_features))
    converters = {k: str for k in conf.selected_categorical_features}
    for f in ordinal_features:
        converters[f] = np.float32
    # Checked that campaign has on 0s and 1s using
    converters['campaign'] = np.int8
    converters['y'] = str
    data = pd.read_csv(conf.data_path, converters=converters)
    # remove dependent variable
    conf.selected_cols.remove('y')
    X = data[conf.selected_cols]

    # convert the two classes into integers
    y = LabelBinarizer().fit_transform(data.y).flatten()

    if conf.only_campaign:
        chosen = X.campaign == 1
        X = X[chosen].reset_index()
        X.drop(['campaign'], axis=1, inplace=True)
        conf.int_features.remove('campaign')
        y = y[chosen]

    if conf.only_non_campaign:
        chosen = X.campaign == 0
        X = X[chosen].reset_index()
        conf.campaign_related_features.append('campaign')
        for c in conf.campaign_related_features:
            if c in X.columns:
                X.drop(c, axis=1, inplace=True)
        for c in conf.campaign_related_features:
            if c in conf.int_features:
                conf.int_features.remove(c)
            if c in conf.selected_categorical_features:
                conf.selected_categorical_features.remove(c)
            if c in ordinal_features:
                ordinal_features.remove(c)
        y = y[chosen]

    X_ordinal = X[ordinal_features]
    X_ints = X[conf.int_features]
    X_cat = X[conf.selected_categorical_features]

    # only apply standardise on ordinal features
    # categorical features will be one hot encoded as number of
    # categories in each categorical feature are only a few
    scalar = standardise_features(X_ordinal)
    # use drop first = True otherwise will result in singular data matrix,
    # and numerical issues with optimisation
    dfs_to_join = [pd.get_dummies(X_cat[c], prefix=c,
                                  drop_first=config.discard_dependent_category)
                   for c in conf.selected_categorical_features]
    X_ordinal_scaled = pd.DataFrame(scalar.transform(X_ordinal),
                                    columns=X_ordinal.columns)
    X_final = pd.concat([X_ordinal_scaled, X_ints] + dfs_to_join, axis=1)

    # check all desired columns are present
    _check_columns_after_dataprep()

    return X_final, y


def _find_best(best_clf, best_score, report, classifier,
               column_names, coefs=None):

    print('Model with {}: {} found'.format(config.grid_search_criteria,
                                           report[config.grid_search_criteria]))
    if best_score < report[config.grid_search_criteria]:
        best_clf = classifier
        best_score = report[config.grid_search_criteria]
        print('====>>>>Best model chosen<<<<=====')

        # coefs help is compare feature importances across models
        if hasattr(best_clf, 'coef_'):
            coefs = {c: cf for c, cf in zip(column_names, best_clf.coef_.flatten())}
            assert len(column_names) == len(classifier.coef_.flatten())
        elif hasattr(best_clf, 'feature_importances_'):
            coefs = {c: cf for c, cf in
                     zip(column_names, best_clf.feature_importances_.flatten())}
            assert len(column_names) == len(
                classifier.feature_importances_.flatten())
        else:
            raise AttributeError('Should not be here')

        # make sure we have expected number of coefficients
        assert len(coefs) == len(column_names)

        return best_clf, best_score, coefs
    else:
        return best_clf, best_score, coefs


def analyze(config, X, y):
    classifier = config.classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        shuffle=True,
                                                        test_size=0.3,
                                                        random_state=2)
    classifier.fit(X_train, y_train)
    print("Before balancing dataset and feature selection")

    # report is classification report
    report = utils.print_clf_scores(classifier, X_test, y_test)

    best_clf = classifier  # this is the only model so far
    best_score = report[config.grid_search_criteria]

    # over weight classifier towards minority class
    print("Before balancing dataset and feture selection: only overweighting "
          "minority class proportional to class ratios")
    C = Counter(y)

    classifier.class_weight = {0: 1, 1: C[0] / C[1]}  # ignored by xgboost
    # only used by xgboost, ignored by others
    classifier.scale_pos_weight = C[0] / C[1]

    print('Classifier weights: ', classifier.class_weight)
    classifier.fit(X_train, y_train)
    report = utils.print_clf_scores(classifier, X_test, y_test)

    best_clf, best_score, coefs = _find_best(best_clf, best_score,
                                             report, classifier,
                                             X_test.columns)

    # much better on class 1, but may be feature selection can improve further
    # make sense to do the gridsearch after balancing dataset
    if config.optimise:
        # If optimise is on, let's use gridsdearch to select best classifier
        # parameters
        print("\n\nRunning gridsearch ....\n\n\n")

        # use original training data in gridserach with a class weight
        # as grid search will internally use cross-validation and validate
        # against oversampled majority class (and not X_test), which is
        # incorrect

        grd_srch = GridSearchCV(
            estimator=classifier, param_grid=config.p_grid,
            scoring=config.scorers,
            refit=config.grid_search_criteria,
            n_jobs=-1, cv=config.crossval_folds, return_train_score=True)
        grd_srch.fit(X_train, y_train)
        # this returns estimator with best x-val score
        classifier = grd_srch.best_estimator_
        print("Scores after grid search:")
        report = utils.print_clf_scores(classifier, X_test, y_test)
        best_clf, best_score, coefs = \
            _find_best(best_clf, best_score, report, classifier,
                       X_test.columns, coefs)

    # It's an imbalanced classification problem
    # Let's balance the data set using imblearn
    # https://github.com/scikit-learn-contrib/imbalanced-learn
    print("================Balancing dataset==================================")
    sampler = config.sampling_algo
    X_tr_bal, y_tr_bal = sampler.fit_sample(X_train, y_train)
    X_tr_bal = pd.DataFrame(data=X_tr_bal, columns=X_train.columns)
    # balanced_clf = make_pipeline(sampler, best_classifier)
    # after balancing, make sure there are no class weights
    classifier.class_weight = {0: 1, 1: 1}  # sklearn classifiers
    classifier.scale_pos_weight = 1  # xgboost
    classifier.fit(X_tr_bal, y_tr_bal)

    report = utils.print_clf_scores(classifier, X_test, y_test)
    best_clf, best_score, coefs = \
        _find_best(best_clf, best_score, report, classifier,
                   X_test.columns, coefs)


    # with feature engineering/selection results may be improved further
    if config.feature_selection:
        print("=======selecting features using RFE============================")
        # feature selection
        # first select using rfe
        if (hasattr(config.classifier, 'coef_') or
                hasattr(config.classifier, 'feature_importances_')):
            selected_cols = select_features_rfe(classifier, X_tr_bal, y_tr_bal)
            print('Selected the following columns based on RFE:\n',
                  selected_cols)
            classifier.fit(X_tr_bal[selected_cols], y_tr_bal)
            report = utils.print_clf_scores(classifier, X_test[selected_cols],
                                            y_test)
            best_clf, best_score, coefs = \
                _find_best(best_clf, best_score, report, classifier,
                           selected_cols, coefs)
        else:
            print("Note: Feature selection using RFE was not possible as "
                  "{} does not support it.".format(
                classifier.__class__.__name__))
            print("Using all features.")
            selected_cols = X_tr_bal.columns
    else:
        selected_cols = X_tr_bal.columns

    config.selected_cols = selected_cols

    # further select based on statistical significance test
    if config.significance_test:
        print("=====Performing significance test of selected features=========")

        # keep checking, and discarding more columns until selected columns and
        # significant columns are the same
        pass_no = 0
        while True:
            pass_no += 1
            try:
                significant_cols = logit_filter(X_tr_bal[selected_cols],
                                                y_tr_bal)
            except np.linalg.LinAlgError as e:
                print(np.linalg.LinAlgError(e))
                print("We encountered np.linalg.LinAlgError error while "
                      "solving the logit paramters")
                print("Improve matrix condition number possibly by removing "
                      "collinear columns")
                break

            print('Selected_cols after {} logit pass(es)'.format(pass_no))
            print(significant_cols)

            if set(selected_cols).__eq__(set(significant_cols)):
                # train and test again
                classifier.fit(X_tr_bal[significant_cols], y_tr_bal)
                report = utils.print_clf_scores(classifier, X_test[
                    significant_cols], y_test)
                best_clf, best_score, coefs = \
                    _find_best(best_clf, best_score, report, classifier,
                               significant_cols, coefs)
                break
            selected_cols = significant_cols[:]

    if config.random_projection:  # test rp and pca
        # not pursued as loses interpretation of features,
        # which is important for the assignment
        assert not config.feature_selection
        from sklearn.decomposition import PCA
        # transformer = GaussianRandomProjection(n_components=30)
        pca = PCA(n_components=10)
        pca.fit(X_tr_bal)
        X_tr_bal_rp = pca.transform(X_tr_bal)
        print(pca.explained_variance_ratio_)
        classifier.fit(X_tr_bal_rp, y_tr_bal)
        print('After random projection')
        report = utils.print_clf_scores(classifier, pca.transform(X_test),
                                        y_test)
        best_clf, best_score, coefs = \
            _find_best(best_clf, best_score, report, classifier,
                       X_test.columns)

    return best_clf, best_score, coefs, X_test, y_test


# test module
if __name__ == '__main__':

    X_final, y = data_prep(config)
    clf, score, coefs, X_test, y_test = analyze(config, X_final, y)

    print(clf)
    print(score)
    print(coefs)
