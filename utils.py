import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve,\
    precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.fixes import signature


def plot_roc(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)[:, 1]
    _roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label=clf.__class__.__name__ + ' (AUC = {:.2f})'.format(
        _roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")


def precision_recall_threshold(clf, X_test, y_test, t=0.5):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """

    # this gives the probability [0,1] that each sample belongs to class 1
    y_scores = clf.predict_proba(X_test)[:, 1]

    # generate new class predictions based on the adjusted_classes
    # function above and view the resulting confusion matrix.
    # generate the precision recall curve
    p, r, thresholds = precision_recall_curve(y_test, y_scores)

    # This adjusts class predictions based on the prediction threshold
    y_pred_adj = y_scores > t

    print(confusion_matrix_df(y_test, y_pred_adj))

    # plot the curve
    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall curve ^ = current threshold")
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.fill_between(r, p, step='post', alpha=0.2,
                     color='b')
    plt.ylim([0, 1.01])
    plt.xlim([0, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # plot the current threshold on the line
    # close_default_clf = np.argmin(np.abs(thresholds - t))
    # plt.plot(r[close_default_clf], p[close_default_clf], '^', c='k',
    #          markersize=15)
    plt.show()


def plot_precision_recall_curve(clf, X_test, y_test):
    if hasattr(clf, 'decision_function'):
        y_scores = clf.decision_function(X_test)
    elif hasattr(clf, 'predict_proba'):
        y_scores = clf.predict_proba(X_test)[:, 1]
    else:
        print('Precision recall curve not possible')
        return
    average_precision = average_precision_score(y_test, y_scores)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))


def adjusted_classes(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold
    (t). Will only work for binary classification problems.
    """
    return y_scores > t


# feature selection helper by inspecting scatter plots
def plot_features_vs_each_other(X, y, c1, c2):
    x1 = X[c1]
    y2 = X[c2]
    plt.scatter(x1, y2, alpha=0.7, c=y)
    plt.xlabel(c1)
    plt.ylabel(c2)


def my_classification_report(y_t, y_p):
    report = classification_report(y_t, y_p, output_dict=True)
    p, r, f, s = precision_recall_fscore_support(y_t, y_p, beta=0.1,
                                                 labels=[1])

    report['1']['fbeta2'] = f[0]
    print('Classification Report:\n', report['1'])
    return report


def confusion_matrix_df(y_t, y_p):
    return pd.DataFrame(confusion_matrix(y_t, y_p),
                        columns=['pred_0', 'pred_1'], index=['0', '1'])


def print_clf_scores(clf, X_t, y_t):
    # predict on the test data
    print('='*80)
    y_p = clf.predict(X_t)
    if hasattr(clf, 'score'):
        print('Accuracy on test set: {:.2f}'.format(clf.score(X_t, y_t)))
    print('Test Set confusion matrix:\n {}'.format(
        confusion_matrix_df(y_t, y_p)))
    report = classification_report(y_t, y_p, output_dict=True)
    # print('Classification report:\n', classification_report(y_t, y_p))
    print('Classification report:\n', my_classification_report(y_t, y_p))
    return report['1']


# def target_score(y_true, y_pred, target_ratio):
#     """
#     modify default score of sklearn classifiers
#
#     This custom score is the ratio of false positive to false negative.
#
#     """
#     cfm = confusion_matrix(y_true, y_pred)
#     score = abs(target_ratio - cfm[0, 1] / cfm[1, 0])
#     print('Target score found: {}'.format(cfm[0, 1] / cfm[1, 0]))
#     return score
#
#
# cfm_ratio_score = make_scorer(target_score, greater_is_better=False,
#                               target_ratio=10)


def plot_precision_recall_vs_threshold(clf, X_test, y_test):

    if hasattr(clf, 'decision_function'):
        y_scores = clf.decision_function(X_test)
    elif hasattr(clf, 'predict_proba'):
        y_scores = clf.predict_proba(X_test)[:, 1]
    else:
        print('Precision recall curve not possible')
        return
    average_precision = average_precision_score(y_test, y_scores)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    """
    Copied from:
    https://www.kaggle.com/kevinarvai/fine-tuning-a-classifier-in-scikit-learn

    Originally..
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.title("Precision and Recall Scores as a function "
              "of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')
