import math as mt
import random
from imblearn.combine import SMOTETomek  # doctest: +NORMALIZE_WHITESPACE
from utils import load_json, save_json
from conf import VALIDATION_NAMES_BY_LOC_PATH


# Reviewed
# Descripción:
# Tomamos val_size como proporcion de pacientes para test
# Calculamos la proporcion por separado para pacientes con y sin epilepsia
# para evitar validar con una sola clase.
# Elegimos nombres random de pacientes mezclando indices
# Retornamos dos listas de pacientes, los que se usan para entrenar y
# testear (model_patients) y los que vamos a usar despues para validar
# Si es la primera vez para esta location, se persiste el val_set en un json.
def pull_apart_validation_set(patients_dic, location, val_size=0.3,
                              balance_types=None):
    print('Pulling apart validation patient set...')
    # Loads predefined validation patient names randomly selected for location
    names_by_loc = load_json(VALIDATION_NAMES_BY_LOC_PATH)
    if location not in names_by_loc.keys():  # first time, we set validation set
        # Build validation set
        validation_names = get_N_fold_bootstrapping_partition(
            [p for p in patients_dic.values()],
            N=1,
            test_size=val_size,
            location=location,
            balance_types=balance_types)[0]

        names_by_loc[location] = validation_names
        save_json(names_by_loc, VALIDATION_NAMES_BY_LOC_PATH)  # Update json

    else:
        print('Loading predefined {0} Validation names'.format(location))
        validation_names = names_by_loc[location]
    print('Validation patient names in {l}: {v}'.format(l=location,
                                                        v=validation_names))
    model_patients = [p for p_name, p in patients_dic.items() if p_name not
                      in validation_names]
    validation_patients = [p for p_name, p in patients_dic.items() if p_name
                           in validation_names]

    return model_patients, validation_patients


# Reviewed
def get_k_fold_random_partition(target_patients, K):
    partition_names = [[] for i in range(K)]
    idx = list(range(len(target_patients)))  # [1,2...N]
    random.shuffle(idx)
    for i in range(len(target_patients)):
        partition_names[i % K].append(target_patients[idx[i]].id)
    return partition_names


# Reviewed
def get_N_fold_bootstrapping_partition(target_patients, N, test_size=0.3,
                                       location='Whole Brain',
                                       balance_types=None):
    '''
    :param target_patients: lists of patients to build train-test folds
    :param N: amount of folds, iterations of cross validation
    :param test_size: proportion of patients for testing each iteration
    :return: N lists of patient names in target_pat to test in a
    # iteration.
    '''
    # We will take val_size of each soz_patients and healthy patients to
    # avoid the possibility of validating with just one class.
    if balance_types is None:
        folds = get_pat_folds(target_patients, test_size)
    else:
        soz_pat, healthy_pat = [], []
        for p in target_patients:
            if p.has_epileptic_activity(location, balance_types):
                soz_pat.append(p)
            elif p.has_nsoz_activity(location, balance_types):
                healthy_pat.append(p)
            else:
                print('Building folds, skipping patient {p}, 0 events'.format(
                    p=p.id))
        soz_folds = get_pat_folds(soz_pat, N, test_size)
        healthy_folds = get_pat_folds(healthy_pat, N, test_size)
        folds = []
        for s, h in zip(soz_folds, healthy_folds):
            fold = s + h
            random.shuffle(fold)
            folds.append(fold)
    return folds


# Reviewed
def get_pat_folds(target_patients, N, test_size):
    partition_names = []
    idx = list(range(len(target_patients)))  # [0,1,2...len-1]
    # Mezcla N veces y forma una lista con los primeros len(patients) *
    # test_size indices despues de mezclar
    for i in range(N):
        random.shuffle(idx)
        index_list = idx[:mt.ceil((len(target_patients) * test_size))]
        names_list = [target_patients[i].id for i in index_list]
        partition_names.append(names_list)

    return partition_names


# Features del modelo de ml segun el hfo_type
def ml_field_names(hfo_type_name, include_coords=False):
    if hfo_type_name in ['RonO', 'Fast RonO']:
        field_names = ['duration', 'freq_pk', 'power_pk']
        ''' YA probe y esto hace que diga q todo es SOZ
        field_names = ['duration',
                       'power_pk',  # pk showed a larger S with kruskal-wallis
                       'power_av',
                       'freq_pk',
                       'freq_av',
                       'slow_angle',
                       'delta_angle',
                       'theta_angle',
                       'spindle_angle',
                       'slow',
                       'delta',
                       'theta',
                       'spindle'
                       ]#
                       '''
    else:
        field_names = ['duration',
                       'power_pk',
                       'freq_pk',
                       ]  # 'spike_angle', 'freq_av',   'power_av'
    if include_coords:
        for c in ['x', 'y', 'z']:
            field_names.append(c)

    return field_names


# Reviewed
def patients_with_more_than(count, patients_dic, hfo_type_name):
    with_hfos = dict()
    without_hfos = dict()
    for p_name, p in patients_dic.items():
        if sum([len(e.events[hfo_type_name]) for e in
                p.electrodes]) > count:  # if has more than count hfos passes
            with_hfos[p_name] = p
        else:
            without_hfos[p_name] = p
    return with_hfos, without_hfos


# Reviewed
def analize_fold_balance(train_labels):
    positive_class = 0
    negative_class = 0
    tot = len(train_labels)
    i = 0
    for l in train_labels:
        i += 1
        if l:
            positive_class += 1
        else:
            negative_class += 1
    print('Fold balance: Positive: {0} Negative: {1}. Count: {2}'.format(
        round(positive_class / tot, 2), round(negative_class / tot, 2), tot))


# Reviewed
def balance_samples(features, labels):
    prev_count = len(features)
    analize_fold_balance(labels)

    print('Performing resample with SMOTETomek...')
    print('Original train hfo count : {0}'.format(prev_count))

    # smt = RepeatedEditedNearestNeighbours( n_jobs=-1)
    # smt = NeighbourhoodCleaningRule(sampling_strategy='majority',
    # n_neighbors=3, n_jobs=-1)
    smt = SMOTETomek(sampling_strategy=1, random_state=42, n_jobs=4)
    features, labels = smt.fit_resample(features, labels)
    post_count = len(features)

    print('{0} instances after SMOTE...'.format(post_count))
    return features, labels
