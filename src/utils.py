import json
import random
import time
from pathlib import Path
import pandas as pd
import numpy as np
import pickle


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def save_json(info_dic, saving_path):
    with open(saving_path, "w") as file:
        json.dump(info_dic, file, indent=4, sort_keys=True)


def load_json(saving_path):
    with open(saving_path) as json_file:
        return json.load(json_file)


def load_object(filename):
    with open('{0}.pkl'.format(filename), 'rb') as input:
        return pickle.load(input)


def save_object(obj, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open('{0}.pkl'.format(filename), 'wb') as output:  # Overwrites any
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


LOG = {}


def log(msg=None, msg_type=None, patient=None, electrode=None):
    print_msg = False
    if print_msg and msg is not None:
        print(msg)

    if msg_type is not None:
        assert (patient is not None)
        if not patient in LOG.keys():
            LOG[patient] = dict()

        if msg_type == 'BLOCK_DURATION':
            if not msg_type in LOG[patient].keys():
                LOG[patient][msg_type] = 0

            LOG[patient][msg_type] += 1
        else:
            assert (electrode is not None)

            if not electrode in LOG[patient].keys():
                LOG[patient][electrode] = dict()

            if not msg_type in LOG[patient][electrode].keys():
                LOG[patient][electrode][msg_type] = 0

            LOG[patient][electrode][msg_type] += 1


def time_counter(callable_code):
    start_time = time.clock()
    res = callable_code()
    print('Runned in {0} minutes'.format(round((time.clock() -
                                                start_time) / 60), 2))
    return res


def constant(f):
    def fset(self, value):
        raise TypeError('You cant modify this constant')

    def fget(self):
        return f()

    return property(fget, fset)


@constant
def FOO():
    return 'constant'


# Randomly maps patient id strings to 1... N
def map_pat_ids(model_patients):
    patient_names = [p.id for p in model_patients]
    random.shuffle(patient_names)
    return [patient_names.index(p.id) + 1 for p in model_patients]


def unique_patients(collection, crit):
    cursor = collection.find(crit)
    docs = set()
    for doc in cursor:
        docs.add(doc['patient_id'])
    patient_ids = list(docs)
    patient_ids.sort()
    print("Unique patients count: {0}".format(len(patient_ids)))
    print(patient_ids)
    return patient_ids


def print_info(info_evts, file):
    # Printing format for saving miscellaneous data in .txt files
    attributes_width = max(len(key) for key in info_evts.keys()) + 2  # padding
    header = '{attr} || {value}'.format(
        attr='Attributes'.ljust(attributes_width), value='Values')
    sep = ''.join(['-' for i in range(len(header))])
    print(header, file=file)
    print(sep, file=file)
    for k, v in info_evts.items():
        k, v = (k, v) if k != 'evt_rates' else ('mean_of_rates', np.mean(v))
        if isinstance(v, list) or k == 'patients_dic_in_loc':  # too long to be
            # printed
            continue
        else:
            row = '{attr} || {value}'.format(attr=k.ljust(attributes_width),
                                             value=str(v))
            print(row, file=file)

    pat_coord_null = info_evts['pat_with_x_null'].union(
        info_evts['pat_with_y_null'])
    pat_coord_null = pat_coord_null.union(info_evts['pat_with_z_null'])
    print('Count of patients with null cords: {0}'.format(len(pat_coord_null)),
          file=file)
    print('List of patients with null cords: {0}'.format(pat_coord_null),
          file=file)

    pat_with_empty_loc = info_evts['pat_with_loc2_empty'].union(
        info_evts['pat_with_loc3_empty'])
    pat_with_empty_loc = pat_with_empty_loc.union(
        info_evts['pat_with_loc5_empty'])
    print(
        'Count of patients with empty loc: {0}'.format(len(pat_with_empty_loc)),
        file=file)
    print('List of patients with empty loc: {0}'.format(pat_with_empty_loc),
          file=file)


def first_key(dic):
    return [k for k in dic.keys()][0]


def print_event_count_per_pat(patients, hfo_type):
    print('Event count of type {0}'.format(hfo_type))
    counts = []
    ids = []
    for pat in patients:
        ids.append(pat.id)
        counts.append(pat.get_types_count([hfo_type]))
    data = pd.DataFrame({'Pat_ID': ids, 'Count': counts})
    print(data.info())
    print(data)


# Reviewed
def hfo_count_quantiles(patients_dic, hfo_type_name):
    qs = [0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4]
    counts = []
    names = []
    for p_name, p in patients_dic.items():
        count = 0
        for e in p.electrodes:
            count += len(e.events[hfo_type_name])

        names.append((p_name, count))
        counts.append(count)
    names.sort(key=lambda x: x[1])
    print(names)
    print('Sorted Patient HFO counts {0}'.format(sorted(counts)))
    quantiles = np.quantile(counts, qs, interpolation='lower')
    # devuelve key:cuantil; value: cuantos eventos necesito tener al
    # menos para ser superior al cuantil de la key, cuantos pacientes quedarian
    # despues del filtro
    return {qs[i]: (quantiles[i], len([c for c in counts if c > quantiles[i]]))
            for i in range(len(qs))}


def get_patient_data(elec_collection, evt_collection, output_csv):
    pipeline = [
        {"$match": {
            "patient_id": {"$in": ['4163', '477', 'IO005', '463', '4100',
                                   '466', '4150', '449', '451', 'IO015',
                                   '2061', '448', '4166', '478', 'IO014',
                                   '468', '456', '4122', '4145', '4110',
                                   'IO006', 'IO010', '475', '458', 'IO013',
                                   'IO009', '481', 'IO004', '480', 'IO008',
                                   '4124', 'IO027', 'IO022', '474', '479',
                                   '473']}
        }
        },
        {"$project": {
            "patient_id": 1,
            "age": 1,
            "gender": 1,
            "riskfactor1": 1,
            "mri1": 1,
            "pet1": 1,
            "sztype1": 1,
            "pathcode1": 1
        }
        },
        {"$group": {
            "_id": "$patient_id",
            "age": {"$addToSet": "$age"},
            "gender": {"$addToSet": "$gender"},
            "riskfactor1": {"$addToSet": "$riskfactor1"},
            "mri1": {"$addToSet": "$mri1"},
            "pet1": {"$addToSet": "$pet1"},
            "sztype1": {"$addToSet": "$sztype1"},
            "pathcode1": {"$addToSet": "$pathcode1"}
        }
        }
    ]

    elec_cur = elec_collection.aggregate(pipeline)
    evt_cur = evt_collection.aggregate(pipeline)

    pat_properties = ['_id', 'age', 'gender',
                      'riskfactor1', 'mri1', 'sztype1', 'pathcode1']
    for cur in [elec_cur, evt_cur]:
        data = dict({k: [] for k in pat_properties})
        for doc in cur:
            for k in pat_properties:
                if k == '_id':
                    data[k].append(doc[k])
                elif len(doc[k]) > 1:
                    raise NotImplementedError('Shouldnt happen')
                else:
                    data[k].append(doc[k][0])
        df = pd.DataFrame(data, columns=pat_properties)
        filename = 'elec.csv' if cur is elec_cur else 'evt.csv'
        out_csv = output_csv + '/' + filename
        print('out path: ', out_csv)
        print('Saving csv...')
        print(df)
        df.to_csv(out_csv)


'''
Sacar pacientes que tienen menos de X eventos, tiende a sacar los fisi
print('All patients count: ')
print_event_count_per_pat([p for p in patients_dic.values()], hfo_type)
quantiles_dic = hfo_count_quantiles(patients_dic, hfo_type)
print('Hfo count quantile dic {0}'.format(quantiles_dic))  # ej
# quantiles_dic[0.3][0]
enough_hfo_pat, excluded_pat = patients_with_more_than(quantiles_dic[
                                                           0.2][0],
                                                       patients_dic,
                                                       hfo_type
)

excluded_healthy_evts, excluded_soz_evts = 0, 0
for p in excluded_pat.values():
    excluded_healthy_evts_i, excluded_soz_evts_i = p.get_classes_weight(
        hfo_type)
    excluded_healthy_evts += excluded_healthy_evts_i
    excluded_soz_evts += excluded_soz_evts_i
excluded_weight = excluded_soz_evts /(excluded_healthy_evts+
                                      excluded_soz_evts)

training_healthy_evts, training_soz_evts = 0, 0
for p in enough_hfo_pat.values():
    training_healthy_evts_i, training_soz_evts_i = p.get_classes_weight(
        hfo_type)
    training_healthy_evts += training_healthy_evts_i
    training_soz_evts += training_soz_evts_i
training_weight = training_soz_evts / (training_healthy_evts +
                                       training_soz_evts)

print('Balance excluded, training: {0}, {1}'.format(excluded_weight, 
training_weight))
'''
