import warnings
import getpass
from random import random, choices

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)
import pydot

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.tree import export_graphviz
from sklearn.svm import SVC

# Machine learning sklearn algorithms
def naive_bayes(train_features, train_labels, test_features, feature_list=None,
                hfo_type_name=None):
    clf = GaussianNB()
    clf.fit(train_features, train_labels)
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]
    return clf_predictions, clf_probs, clf


def svm_m(train_features, train_labels, test_features, feature_list=None,
          hfo_type_name=None):
    # kernel = [linear, poly, rbf, sigmoid]
    kernel = 'linear'
    # clf = svm.SVC(kernel=kernel, C=1, probability=True, degree= 3,
    # gamma='auto')
    clf = CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000))
    clf.fit(train_features, train_labels)

    clf_predictions = clf.predict(test_features)
    if hasattr(clf, "predict_proba"):
        clf_probs = clf.predict_proba(test_features)[:, 1]
    else:
        clf_probs = None

    return clf_predictions, clf_probs, clf


def xgboost(train_features, train_labels, test_features, feature_list=None,
            hfo_type_name=None):
    clf = XGBClassifier(nthread=-1)
    clf_predictions = clf.predict(test_features)
    clf_probs = clf.predict_proba(test_features)[:, 1]

    # graphics.feature_importances(feature_list, clf.feature_importances_,
    # hfo_type_name, fig_id)
    return clf_predictions, clf_probs, clf


# SIMULATOR

# Reviewed
def simulator(test_labels, distr, confidence):
    simulated_preds = []
    simulated_probs = []
    for soz_label in test_labels:
        r = random()
        if soz_label:
            if r <= confidence:
                pred = 1
                prob = choices(distr['TP'],
                               weights=[1 / len(distr['TP'])] * len(
                                   distr['TP']), k=1)[0]
            else:
                pred = 0
                prob = choices(distr['FN'],
                               weights=[1 / len(distr['FN'])] * len(
                                   distr['FN']), k=1)[0]
        else:
            if r <= confidence:
                pred = 0
                prob = choices(distr['TN'],
                               weights=[1 / len(distr['TN'])] * len(
                                   distr['TN']), k=1)[0]
            else:
                pred = 1
                prob = choices(distr['FP'],
                               weights=[1 / len(distr['FP'])] * len(
                                   distr['FP']), k=1)[0]

        assert (
            not isinstance(prob, list))  # choices returns a list of k values
        simulated_preds.append(pred)
        simulated_probs.append(prob)
    return simulated_preds, simulated_probs


def generate_trees(feature_list, train_features, train_labels,
                   amount=1,
                   directory='/home/{user}'.format(user=getpass.getuser())):
    # Limit depth of tree to 3 levels
    rf_small = RandomForestClassifier(n_estimators=amount, max_depth=4)
    rf_small.fit(train_features, train_labels)
    for i in range(amount):
        # Extract the small tree
        tree_small = rf_small.estimators_[i]
        # Save the tree as a png image
        out_path = '{dir}/thesis_tree_{k}.dot'.format(dir=directory, k=i)
        export_graphviz(tree_small,
                        out_file=out_path,
                        feature_names=feature_list,
                        rounded=True,
                        precision=1)
        (graph,) = pydot.graph_from_dot_file(out_path)
        graph.write_png('{dir}/thesis_tree_{k}.png'.format(dir=directory, k=i))


CLASSIFIERS = {
    #'Linear SVC': CalibratedClassifierCV(LinearSVC(C=1, max_iter=10000)),
    'SGD': SGDClassifier(loss='log', penalty='l2', max_iter=1000, n_jobs=-1),
    # loss='log',  # 'modified_huber'
    # penalty='elasticnet',
    # max_iter=1000, n_jobs=-1),
    'Logistic Regression': LogisticRegression(
            penalty='l2',  # 'l1' 'elasticnet'
            random_state=42,
            n_jobs=-1, max_iter=1000),
    #'SVC': SVC(C=1.0,
    #           kernel='rbf',
    #           degree=3,
    #           probability=True,
    #           tol=0.001,
    #           class_weight='balanced',
    #           max_iter=5000,
    #           random_state=None)

}

'''
'XGBoost': XGBClassifier(
        learning_rate=0.05,
        n_estimators=500,  # 100
        max_depth=6,
        objective='binary:logistic',
        nthread=-1,
        seed=10,
    ),
    'Logistic Regression': LogisticRegression(
            penalty='l2', #'l1' 'elasticnet'
            random_state=42,
            n_jobs=-1)

'''
################################################################################
