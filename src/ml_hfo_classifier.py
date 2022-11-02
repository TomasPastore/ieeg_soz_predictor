import copy
from pathlib import Path
import pandas as pd
import numpy as np
import math as mt

import graphics
from conf import CLASSIFIERS
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV

from ml_hfo_rate_filter import get_soz_confidence_thresh, phfo_thresh_filter
from partition_builder import get_N_fold_bootstrapping_partition
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn import preprocessing, metrics
from partition_builder import pull_apart_validation_set, ml_field_names
from soz_predictor import region_info
from utils import load_object, save_object
from db_parsing import WHOLE_BRAIN_L0C


# Returns best estimator for validation in location
def ml_hfo_classifier_train(patients_dic, location, hfo_type, use_coords,
                            model_name, sim_recall=None, saving_dir=None):
    '''
    Parameters:
    patients_dic: tiene como claves los patient_id y como valor la data
    location: es la region en la cual se hace ml
    hfo_type: es el tipo al cual le aplicaremos,
    use_coords: indica si usar x,y,z como features en ml o no.
    '''
    print('\nML HFO classifier training...')
    print('Location ---> {0}'.format(location))
    print('HFO type ---> {0}'.format(hfo_type))
    print('Model name ---> {0}'.format(model_name))
    print('Saving directory: {0}'.format(saving_dir))

    # Optional: pickle cache for patients data
    pat_dic_filename = str(Path(saving_dir, 'pat_dic_pickles', '{l}_{'
                                                               't}_pat_dic'.format(
        l=location, t=hfo_type).replace(' ', '_')))
    save_object(patients_dic, pat_dic_filename)

    # Preparing data groups
    model_patients, validation_patients = pull_apart_validation_set(
        patients_dic,
        location,
        val_size=0.3,
        balance_types=[hfo_type])

    folds_test_names = get_N_fold_bootstrapping_partition(
        model_patients,
        N=100,
        test_size=0.3,
        location=location,
        balance_types=[hfo_type])

    field_names = ml_field_names(hfo_type, include_coords=use_coords)
    X_train, y_train, groups, f_names = serialize_patients_to_events(
        model_patients,
        hfo_type,
        field_names)

    # Choosing metrics
    scoring = {
        'roc_auc': 'roc_auc',
        'recall': 'recall',
        'f1': 'f1',
        'tp': metrics.make_scorer(tp),
        'tn': metrics.make_scorer(tn),
        'fp': metrics.make_scorer(fp),
        'fn': metrics.make_scorer(fn),
    }
    # Chossing cross validation algorithm
    cv_start = {clf_name: pats_folds_to_obs_folds(folds_test_names,
                                                  model_patients,
                                                  groups) for clf_name in
                CLASSIFIERS.keys()}
    # cv_start = {clf_name: list(GroupShuffleSplit(n_splits=100, test_size=.3,
    #                                             random_state=42).split(
    #    X_train, y_train, groups))
    #    for clf_name in CLASSIFIERS.keys()}

    classifiers = copy.deepcopy(CLASSIFIERS)
    cv_with_metrics(classifiers, X_train, y_train, groups, cv_start, scoring)

    # Fine Tuning

    cv_tune = pats_folds_to_obs_folds(folds_test_names,
                                      model_patients,
                                      groups)
    # cv_tune = GroupShuffleSplit(n_splits=100, test_size=.3, random_state=42)
    # cv_tune = list(cv_tune.split(X_train, y_train, groups))
    name = model_name
    CLASSIFIERS_TO_TUNE = {name: copy.deepcopy(classifiers[name])}
    grids = dict()
    if name == 'SGD':
        grids[name] = {
            'loss': ['log'],
            'penalty': ['l2'],
            'alpha': [1e-4, 1e-2, 1e0],
            'max_iter': [1000, 2000],  # number of epochs
            'class_weight': [{1: .6, 0: .4},
                             {1: .7, 0: .3},
                             None],
            'shuffle': [False],
            'learning_rate': ['optimal'],
            # 'early_stopping': [False, True],
            # 'n_iter_no_change': [50],
            'n_jobs': [-1],
        }

        sgd_grid = {
            'loss': ['log'],
            'penalty': ['l2'],
            'alpha': [1e-4, 1e-2, 1e0],
            'max_iter': [1000, 2000],  # number of epochs
            'class_weight': [{1: .6, 0: .4},
                             {1: .7, 0: .3},
                             None],
            'shuffle': [False],
            'learning_rate': ['optimal'],
            'n_jobs': [-1],
        }
    else:
        raise ValueError(
            'Undefined hyperparameter grid for model {0}'.format(name))

    best_params = dict()
    for name, clf in CLASSIFIERS_TO_TUNE.items():
        best_params[name] = fine_tune(X_train, y_train, clf, name,
                                      grids[name], cv_tune,
                                      scoring, objective='recall')
    best_clf = copy.deepcopy(CLASSIFIERS['SGD'])
    best_clf.set_params(**best_params['SGD'])

    X_test, y_test, groups, f_names = serialize_patients_to_events(
        validation_patients, hfo_type, field_names)

    ml_data = dict(
        model_name=model_name,
        best_clf=best_clf,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    return validation_patients, ml_data


def ml_hfo_classifier_validate(hfo_type, location, validation_patients,
                               ml_data, saving_dir):
    print('Validating Model...')
    print('HFO type ---> {0}'.format(hfo_type))
    print('Location ---> {0}'.format(location))
    print('Model name ---> {0}'.format(ml_data['model_name']))

    clf = ml_data['best_clf']
    print('Estimator params: ', clf.get_params())
    model_name = ml_data['model_name']
    X_train, y_train = ml_data['X_train'], ml_data['y_train']
    X_test, y_test = ml_data['X_test'], ml_data['y_test']

    print('Calculating baseline rates for validation set...')
    plot_data_by_loc = {location: dict()}
    validation_pat_dic = {p.id: p for p in validation_patients}
    ri_loc = location if location != WHOLE_BRAIN_L0C else None
    loc_info = region_info(validation_pat_dic,
                           event_types=[hfo_type],
                           flush=True,
                           location=ri_loc,
                           )
    fig_model_name = 'Validation baseline'
    plot_data_by_loc[location][fig_model_name] = loc_info

    print('Testing validation set...')
    pipeline_clf = make_pipeline(preprocessing.StandardScaler(), clf)
    pipeline_clf.fit(X_train, y_train)
    # metrics.plot_roc_curve(pipeline_clf, X_test, y_test)
    # plt.show()
    probas = pipeline_clf.predict_proba(X_test)[:, 1]

    print('Saving probas in validation patients dic')

    def save_probas(validation_patients, probas, hfo_type):
        i = 0
        for p in validation_patients:
            for e in p.electrodes:
                for h in e.events[hfo_type]:
                    h.info['proba'] = probas[i]
                    i = i + 1
        assert (i == len(probas))
        return validation_patients

    annotated_val_patients = save_probas(validation_patients, probas, hfo_type)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas)

    print('Calculating filters ROCs...')
    for tol_fpr in [0.2, 0.8]:  # 0.4,0.4, 0.6,
        thresh = get_soz_confidence_thresh(fpr, tpr, thresholds,
                                           tolerated_fpr=tol_fpr)
        filtered_pat_dic = phfo_thresh_filter(annotated_val_patients,
                                              hfo_type_name=hfo_type,
                                              thresh=thresh,
                                              model_name=model_name)
        loc_info = region_info(filtered_pat_dic,
                               event_types=[hfo_type],
                               flush=True,  # el flush es importante
                               location=ri_loc,
                               )  # calcula la info para la roc con la prob
        # asociada al fpr tolerado
        fig_model_name = 'Thresh: {t} FPR: {f}'.format(t=round(thresh, 3),
                                                       f=tol_fpr)
        plot_data_by_loc[location][fig_model_name] = loc_info

    # YOUDEN
    youden_thresh = youden(fpr, tpr, thresholds)
    print('Youden thresh ---> {0}'.format(youden_thresh))
    filtered_pat_dic = phfo_thresh_filter(annotated_val_patients,
                                          hfo_type_name=hfo_type,
                                          thresh=youden_thresh,
                                          model_name=model_name)
    loc_info = region_info(filtered_pat_dic,
                           event_types=[hfo_type],
                           flush=True,  # el flush es importante
                           location=ri_loc,
                           )  # calcula la info para la roc con la prob
    # asociada al fpr tolerado
    fig_model_name = 'Thresh: {t} FPR: {f}'.format(f='Youden',
                                                   t=round(youden_thresh, 3))
    plot_data_by_loc[location][fig_model_name] = loc_info

    print('Plotting...')
    saving_path = str(Path(saving_dir, '{l}/{t}/{t}_{l}_filters'.format(
        l=location, t=hfo_type).replace(' ', '_')))
    graphics.event_rate_by_loc(plot_data_by_loc,
                               metrics=['pse', 'auc'],
                               title='SGD filters for {t} in {l}'.format(
                                   t=hfo_type, l=location),
                               roc_saving_path=saving_path,
                               partial_roc=False)


# AUX methods

def pats_folds_to_obs_folds(folds_test_names, model_patients, groups):
    # Transforms pat names fold to pat indexes folds
    for fold_test_names in folds_test_names:
        train_idx, test_idx = [], []
        for i, pat_idx in enumerate(groups):  # index in MODEL_PATIENT
            if model_patients[pat_idx].id in fold_test_names:
                test_idx.append(i)
            else:
                train_idx.append(i)
        yield train_idx, test_idx


def serialize_patients_to_events(model_patients, hfo_type, field_names):
    X, y, groups = [], [], []
    pac = [f for f in field_names if 'angle' in f or 'vs' in f]
    for i, p in enumerate(model_patients):
        for e in p.electrodes:
            for h in e.events[hfo_type]:
                if all([isinstance(h.info[f], float) for f in
                        pac]):  # I use
                    # this event only if all the pac is not null, else skip,
                    # if you don't use any '_angle' or 'vs' PAC property this
                    # takes
                    # every event
                    feature_row_i = {}
                    for feature_name in field_names:
                        if 'angle' in feature_name or 'vs' in feature_name:
                            feature_row_i[
                                'SIN({0})'.format(feature_name)] = mt.sin(
                                h.info[feature_name])
                            feature_row_i[
                                'COS({0})'.format(feature_name)] = mt.cos(
                                h.info[feature_name])
                        else:
                            feature_row_i[feature_name] = float(h.info[
                                                                    feature_name])
                    X.append(feature_row_i)
                    assert (e.soz == h.info['soz'])
                    y.append(float(e.soz))
                    groups.append(i)

    X_pd = pd.DataFrame(X)
    feature_names = X_pd.columns  # adds sin and cos for PAC
    return X_pd.values, np.array(y), np.array(groups), feature_names


def tn(y, y_pred): return metrics.confusion_matrix(y, y_pred)[0, 0];


def fp(y, y_pred): return metrics.confusion_matrix(y, y_pred)[0, 1];


def fn(y, y_pred): return metrics.confusion_matrix(y, y_pred)[1, 0];


def tp(y, y_pred): return metrics.confusion_matrix(y, y_pred)[1, 1];


def npv(y, y_pred): return tn(y, y_pred) / (tn(y, y_pred) + fn(y, y_pred))


# NPV: que porcentaje de lo q estoy filtrando es correcto. tiene que ser
# mayor a 0.5 si esta balanceado

def cv_with_metrics(CLASSIFIERS, X, y, groups, cv=None, scoring=None):
    for clf_name, clf in CLASSIFIERS.items():
        print('Crossvalidating with {0}'.format(clf_name))
        print(clf.get_params())
        pipeline_clf = make_pipeline(preprocessing.StandardScaler(), clf)
        cv_i = cv[clf_name]

        # print('\nML training X')
        # print(X)

        cv_results = cross_validate(pipeline_clf, X, y=y, groups=groups,
                                    scoring=scoring, cv=cv_i, n_jobs=-1)
        print('Scoring mean across folds for {0}...'.format(clf_name))
        for key, v in cv_results.items():
            if 'test' in key:
                print('\t{k} --> {m}'.format(k=key, m=np.mean(cv_results[key])))


def fine_tune(X, y, classifier, model_name, grid, cv, scoring, objective):
    print('Fine tuning {0}. Objective: {1}'.format(model_name, objective))
    grid_search = GridSearchCV(estimator=classifier, param_grid=grid, n_jobs=-1,
                               cv=cv, scoring=scoring, refit=objective)
    grid_result = grid_search.fit(X, y)
    print("Best score was : %f using %s" % (
        grid_result.best_score_, grid_result.best_params_))
    return grid_result.best_params_


def assert_folds_ok(folds_test_names, model_patients, groups, hfo_type):
    new_groups = pats_folds_to_obs_folds(folds_test_names, model_patients,
                                         groups)
    pat_ids = groups
    for train, test in new_groups:
        pat_counts = [0] * len(model_patients)
        for t in train:
            # none pat idx is in the other and sum for each pat is eq to
            # group_id
            # sumo o resto y despues miro q el modulo sea igual al count de
            # eventos del paciente
            pat_idx = pat_ids[t]
            pat_counts[pat_idx] += 1  # train
        for t in test:
            pat_idx = pat_ids[t]
            pat_counts[pat_idx] -= 1  # test

        for i, p in enumerate(model_patients):
            assert (abs(pat_counts[i]) == p.get_types_count([hfo_type]))
            if abs(pat_counts[i]) != p.get_types_count([hfo_type]):
                print('el paciente {0} tiene un count de {1} pero el '
                      'metodo de pat dice {2}'.format(i, pat_counts[i],
                                                      p.get_types_count(
                                                          [hfo_type])))


def youden(fpr, tpr, thresholds):
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx]


def boost_train(location, hfo_type, saving_dir):
    pat_dic_filename = str(Path(saving_dir, 'pat_dic_pickles', '{l}_{'
                                                               't}_pat_dic'.format(
        l=location, t=hfo_type).replace(' ', '_')))
    patients_dic = load_object(pat_dic_filename)
    ml_hfo_classifier_train(patients_dic,
                            location=location,
                            hfo_type=hfo_type,
                            use_coords=False,
                            saving_dir=saving_dir)
