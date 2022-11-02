import os
from pathlib import Path
from scipy.stats import mannwhitneyu
from patient import Patient
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from conf import FIG_SAVE_PATH, ML_MODELS_TO_RUN
from db_parsing import (get_granularity, load_patients,
                        EVENT_TYPES, HFO_TYPES, WHOLE_BRAIN_L0C)

from utils import print_info, load_json, save_json
import graphics
import warnings

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)


# 3) Predicting SOZ with rates: Baselines  #####################################
# Returns the data for the ml and the data to plot the ROC baselines
def evt_rate_soz_pred_baseline_localized(elec_collection,
                                         evt_collection,
                                         intraop=False,
                                         load_untagged_coords_from_db=True,
                                         load_untagged_loc_from_db=True,
                                         restrict_to_tagged_coords=True,
                                         restrict_to_tagged_locs=True,
                                         evt_types_to_load=EVENT_TYPES,
                                         subtypes_to_load=None,
                                         evt_types_to_cmp=None,
                                         locations=None,
                                         saving_dir=
                                         FIG_SAVE_PATH[3]['dir'],
                                         models_to_run=ML_MODELS_TO_RUN,
                                         return_pat_dic_by_loc=False,
                                         plot_rocs=False,
                                         remove_elec_artifacts=False):
    if locations is None:
        locations = {0: [WHOLE_BRAIN_L0C]}
    if evt_types_to_cmp is None:
        evt_types_to_cmp = [[t] for t in
                            HFO_TYPES] + [
                               ['Spikes',
                                'Sharp Spikes']]
    print('SOZ predictor Analysis...')
    print('Intraop: {intr}'.format(intr=intraop))
    print('load_untagged_coords_from_db: {0}'.format(
        load_untagged_coords_from_db))
    print('load_untagged_loc_from_db: {0}'.format(load_untagged_loc_from_db))
    print('restrict_to_tagged_coords: {0}'.format(restrict_to_tagged_coords))
    print('restrict_to_tagged_locs: {0}'.format(restrict_to_tagged_locs))
    print('evt_types_to_load : {0}'.format(evt_types_to_load))
    print('evt_to_cmp: {0}'.format(evt_types_to_cmp))
    print('locations: {0}'.format(locations))
    print('saving_dir: {0}'.format(saving_dir))
    print('models_to_run: {0}'.format(models_to_run))

    # Load data
    patients_by_loc = load_patients(
        elec_collection, evt_collection,
        intraop,
        loc_granularity=0,
        locations=[WHOLE_BRAIN_L0C],
        event_type_names=evt_types_to_load,
        subtypes_from_db=subtypes_to_load,
        models_to_run=models_to_run,
        load_untagged_coords_from_db=load_untagged_coords_from_db,
        load_untagged_loc_from_db=load_untagged_loc_from_db,
        restrict_to_tagged_coords=restrict_to_tagged_coords,
        restrict_to_tagged_locs=restrict_to_tagged_locs,
        remove_elec_artifacts=remove_elec_artifacts
    )

    data_by_loc = dict()  # Saves data for returning baseline info by loc
    for granularity, locs in locations.items():
        event_type_data_by_loc = dict()  # For plotting this granularity level
        for loc_name in locs:
            patients_dic = patients_by_loc[WHOLE_BRAIN_L0C]
            file_saving_path = str(Path(saving_dir,
                                        'metadata/loc_{g}'.format(
                                            g=granularity),
                                        loc_name,
                                        loc_name +
                                        '_metadata_sleep_info.txt')).replace(
                ' ', '_')
            Path(file_saving_path).parent.mkdir(parents=True, exist_ok=True)

            with open(file_saving_path, "w+") as file:
                print('Data in {0} for types to compare... '.format(loc_name),
                      file=file)
                for event_type_names in evt_types_to_cmp:
                    if 'Spikes' in event_type_names:
                        type_group_name = 'Spikes'  # merged spikes + sharp
                        # spikes
                    elif len(event_type_names) > 1:  # merged HFO types
                        type_group_name = 'HFOs'
                    else:  # individual HFO types
                        type_group_name = '+'.join(event_type_names)
                    print('\nInfo from: {n}'.format(n=type_group_name),
                          file=file)

                    loc_info = region_info(patients_dic, event_type_names,
                                           location=loc_name if loc_name !=
                                                                WHOLE_BRAIN_L0C
                                           else None)

                    min_pat_count_in_location = 12  # Suggested by Shennan
                    min_pat_with_epilepsy_in_location = 3
                    if loc_info['patient_count'] >= min_pat_count_in_location \
                            and loc_info[
                        'patients_with_epilepsy'] >= \
                            min_pat_with_epilepsy_in_location:
                        print('Files_saving_path {0}'.format(file_saving_path))

                        if loc_name not in data_by_loc.keys():
                            event_type_data_by_loc[loc_name] = dict()
                            data_by_loc[loc_name] = dict()

                        # For ROCs plot
                        event_type_data_by_loc[loc_name][type_group_name] = \
                            loc_info

                        # Saving baseline data
                        data_by_loc[loc_name][type_group_name] = loc_info

                        # Generate de metadata txt file
                        print_info(loc_info, file=file)

                        # For exp 2 stats with hfo_rate
                        data_by_loc[loc_name][type_group_name + '_rates'] = \
                            dict(soz=loc_info['soz_rates'],
                                 nsoz=loc_info['nsoz_rates'])

                        # For 3.ii table
                        data_by_loc[loc_name]['PSE'] = loc_info['pse']
                        data_by_loc[loc_name][type_group_name + '_AUC'] = \
                            loc_info['AUC_ROC']

                        # For saving a location dictionary
                        # Same for each hfo type, whole dic
                        if return_pat_dic_by_loc:
                            data_by_loc[loc_name]['patients_dic'] = \
                                loc_info['patients_dic_in_loc']

                    else:
                        print('Region and type excluded due to lack of '
                              'data --> {0} {1}'.format(loc_name,
                                                        type_group_name))
        if plot_rocs:
            graphics.event_rate_by_loc(event_type_data_by_loc,
                                       metrics=['pse', 'pnee', 'auc'],
                                       roc_saving_path=str(Path(saving_dir,
                                                                'loc_{'
                                                                'g}'.format(
                                                                    g=granularity),
                                                                'rate_baseline')),
                                       change_tab_path=True,
                                       partial_roc=False)
    return data_by_loc


# Gathers info about patients rate data for the types included in the list
# If loc is None all the dic is considered, otherwise only the location asked
def region_info(patients_dic, event_types=EVENT_TYPES, flush=False,
                conf=None, location=None):
    print('Region info location {0}, types {1}.'.format(location, event_types))
    patients_with_epilepsy = set()
    elec_count_per_patient = []
    elec_x_null, elec_y_null, elec_z_null = 0, 0, 0
    elec_cnt_loc2_empty, elec_cnt_loc3_empty, elec_cnt_loc5_empty = 0, 0, \
                                                                    0

    pat_with_x_null, pat_with_y_null, pat_with_z_null = set(), set(), set()

    pat_with_loc2_empty, pat_with_loc3_empty, pat_with_loc5_empty = set(), \
                                                                    set(), \
                                                                    set()

    soz_elec_count, elec_with_evt_count, event_count = 0, 0, 0
    counts = {type: 0 for type in event_types}
    event_rates, soz_labels, soz_rates, nsoz_rates = [], [], [], []

    table_features = ['HFO rate', 'Freq Pk', 'Power Pk', 'Duration']
    properties_stats = {prop: {'soz': [], 'nsoz': []} for prop in
                        table_features}
    if location is not None:
        patients_dic = {p_name: p for p_name, p in patients_dic.items() if \
                        p.has_elec_in(loc=location)}
    pat_in_loc = dict()
    patient_names = []
    montage_df_vec = []
    get_montage_stats = False

    for p_name, p in patients_dic.items():
        patient_names.append(p_name)
        if location is None:
            electrodes = p.electrodes
            pat_in_loc[p_name] = p  # was commented
        else:
            electrodes = [e for e in p.electrodes if getattr(e,
                                                             'loc{i}'.format(i=
                                                             get_granularity(
                                                                 location)))
                          == location]
            pat_in_loc[p_name] = Patient(p_name, p.age, p.gender, electrodes)

        elec_count_per_patient.append(len(electrodes))
        assert (len(electrodes) > 0)
        for e in electrodes:
            if flush:
                e.flush_cache(event_types)
            if e.soz:
                patients_with_epilepsy.add(p_name)
                soz_elec_count = soz_elec_count + 1
            if e.x is None:
                elec_x_null += 1
                pat_with_x_null.add(p_name)
            if e.y is None:
                elec_y_null += 1
                pat_with_y_null.add(p_name)
            if e.z is None:
                elec_z_null += 1
                pat_with_z_null.add(p_name)
            if e.loc2 == 'empty':
                elec_cnt_loc2_empty += 1
                pat_with_loc2_empty.add(p_name)
            if e.loc3 == 'empty':
                elec_cnt_loc3_empty += 1
                pat_with_loc3_empty.add(p_name)
            if e.loc5 == 'empty':
                elec_cnt_loc5_empty += 1
                pat_with_loc5_empty.add(p_name)

            elec_evt_count = e.get_events_count(event_types)
            elec_with_evt_count = elec_with_evt_count + 1 if elec_evt_count > \
                                                             0 else \
                elec_with_evt_count
            event_count += elec_evt_count

            for type in event_types:
                counts[type] += e.get_events_count([type])

            evt_rate = e.get_events_rate(event_types)  # Measured in events/min
            event_rates.append(evt_rate)
            soz_labels.append(e.soz)
            if get_montage_stats:
                for evt_type in HFO_TYPES:
                    for evt in e.events[evt_type]:
                        montage_df_vec.append({
                            'hfo_type': evt.info['type'],
                            'montage': evt.info['montage'],
                            'power_pk': evt.info['power_pk']
                        })
            if e.soz:
                soz_rates.append(evt_rate)
                for t in event_types:
                    for evt in e.events[t]:
                        assert (evt.info['soz'])
                        properties_stats['Freq Pk']['soz'].append(evt.info[
                                                                      'freq_pk'])
                        properties_stats['Power Pk']['soz'].append(evt.info[
                                                                       'power_pk'])
                        properties_stats['Duration']['soz'].append(evt.info[
                                                                       'duration'])

            else:  # NSOZ
                nsoz_rates.append(evt_rate)
                for t in event_types:
                    for evt in e.events[t]:
                        assert not evt.info['soz']
                        properties_stats['Freq Pk']['nsoz'].append(evt.info[
                                                                       'freq_pk'])
                        properties_stats['Power Pk']['nsoz'].append(evt.info[
                                                                        'power_pk'])
                        properties_stats['Duration']['nsoz'].append(evt.info[
                                                                        'duration'])

    elec_count = sum(elec_count_per_patient)
    pse = soz_elec_count / elec_count if elec_count != 0 else 0  # proportion of
    # soz
    # electrodes
    non_empty_elec_prop = elec_with_evt_count / elec_count if elec_count != 0 \
        else 0
    try:
        fpr, tpr, threshold = roc_curve(soz_labels, event_rates)

        pAUC = roc_auc_score(soz_labels, event_rates,
                             max_fpr=0.05)
        auc_roc = roc_auc_score(soz_labels, event_rates)
    except ValueError:
        fpr = tpr = threshold = None
        pAUC = None
        auc_roc = None

    print_patient_names = True
    if print_patient_names:
        print('Region info: patient names: {0}'.format(patient_names))

    age_gender_csv = False
    if age_gender_csv:
        age, gender = [], []
        loc3_by_pat_soz = dict()
        for pat in patient_names:
            p = patients_dic[pat]
            # p.print_age_gender()
            for e in p.electrodes:
                if e.soz:
                    if p.id not in loc3_by_pat_soz.keys():
                        loc3_by_pat_soz[p.id] = set()
                    loc3_by_pat_soz[p.id].add(e.loc3)
            age.append(p.age)
            gender.append(p.gender)
        age_gen = pd.DataFrame({'ID': patient_names, 'Gender': gender, 'Age': \
            age}).sort_values(by=['Gender'])
        output_csv = str(Path(FIG_SAVE_PATH[2]['dir'], 'age_gender.csv'))
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        age_gen.to_csv(output_csv, mode='a',
                       header=not os.path.exists(output_csv))

    if get_montage_stats:
        montage_df = pd.DataFrame(montage_df_vec)
        montage_stat_df_vec = []
        for e_type in HFO_TYPES:
            type_df = montage_df[montage_df['hfo_type'] == e_type]
            montage_0_power = type_df[type_df['montage'] == 0]['power_pk']
            montage_1_power = type_df[type_df['montage'] == 1]['power_pk']
            U, pval = mannwhitneyu(montage_0_power, montage_1_power)
            montage_stat_df_vec.append({
                'hfo_type': e_type,
                'U': U,
                'pval': pval,
                'montage_0_mean': montage_0_power.mean(),
                'montage_1_mean': montage_1_power.mean(),
            })
            print('PVAL', pval)
        montage_stat_df = pd.DataFrame(montage_stat_df_vec)

        if event_types[0] == 'RonO':
            output_csv_stats = str(Path(FIG_SAVE_PATH[2]['dir'],
                                        'montage_stats.csv'))
            output_csv_raw = str(
                Path(FIG_SAVE_PATH[2]['dir'], 'montage_data.csv'))
            Path(output_csv_stats).parent.mkdir(parents=True, exist_ok=True)
            Path(output_csv_raw).parent.mkdir(parents=True, exist_ok=True)
            montage_stat_df.to_csv(output_csv_stats,
                                   header=not os.path.exists(output_csv_stats))
            montage_df.to_csv(output_csv_raw,
                              header=not os.path.exists(output_csv_raw))

    # LOC3 SOZ UNIQUE TABLE
    get_loc3_soz = False
    if location is None and get_loc3_soz:  # None
        ids_col = [[p] * len(loc3_by_pat_soz[p]) for p in patient_names]
        loc_col = [list(loc3_by_pat_soz[p]) for p in patient_names]
        loc3_df = pd.DataFrame({'ID': list(np.concatenate(ids_col)),
                                'LOC_3': list(np.concatenate(loc_col))
                                }).sort_values(by=['ID', 'LOC_3'])
        output_csv = str(Path(FIG_SAVE_PATH[2]['dir'], 'soz_unique_loc3.csv'))
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        loc3_df.to_csv(output_csv,
                       header=not os.path.exists(output_csv))  # mode='a',

    properties_stats['HFO rate']['soz'] = soz_rates
    properties_stats['HFO rate']['nsoz'] = nsoz_rates

    table_type = event_types[0]  # representant, the first type in the type
    # groups
    table_loc = 'Whole Brain' if location is None else location
    save_means = True
    if len(event_types) <= 2 and save_means:
        for soz_str_low in ['soz', 'nsoz']:
            output_csv = str(
                Path(FIG_SAVE_PATH[2]['dir'], soz_str_low + '_mean_table.csv'))
            soz_upper = soz_str_low.upper()
            if table_type != 'Spikes':
                df = pd.DataFrame({
                    'HFO type': table_type,
                    'Location': table_loc,
                    'Property': table_features,
                    '{} mean'.format(soz_upper): [
                        round(np.mean(properties_stats[p][soz_str_low]), 2) for
                        p in
                        table_features],
                    '{} median'.format(soz_upper): [
                        round(np.median(properties_stats[p][soz_str_low]), 2)
                        for p in
                        table_features],
                },
                    columns=['HFO type', 'Location', 'Property',
                             '{} mean'.format(soz_upper),
                             '{} median'.format(soz_upper)])
            else:
                rate = 'HFO rate'
                df = pd.DataFrame({
                    'HFO type': table_type,
                    'Location': table_loc,
                    'Property': [rate],  # only rate for spikes
                    '{} mean'.format(soz_upper): [
                        round(np.mean(properties_stats[rate][soz_str_low]), 2)],
                    '{} median'.format(soz_upper): [
                        round(np.median(properties_stats[rate][soz_str_low]),
                              2)],
                }, columns=['HFO type', 'Location', 'Property',
                            '{} mean'.format(soz_upper),
                            '{} median'.format(soz_upper)])
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, mode='a',
                      header=not os.path.exists(output_csv))
    else:
        print('Did not save means in soz predictor')
    # ROC info for shennan csv
    get_roc_json = True
    if tpr is not None and get_roc_json:
        output_roc_data_json = str(
            Path(FIG_SAVE_PATH[2]['dir'], 'roc_data.json'))
        Path(output_roc_data_json).parent.mkdir(parents=True, exist_ok=True)
        try:
            roc_data = load_json(output_roc_data_json)
        except Exception:
            print('Creating empty json in that path..')
            save_json(dict(), output_roc_data_json)
            roc_data = load_json(output_roc_data_json)

        if table_type not in roc_data.keys():
            roc_data[table_type] = dict()
        roc_data[table_type][table_loc] = {
            'ROC_tpr': list(tpr),
            'ROC_fpr': list(fpr),
            'ROC_thresholds': list(threshold),
            'ROC_AUC': auc_roc,
            'ROC_pAUC_0.05_fpr': pAUC
        }
        save_json(roc_data, output_roc_data_json)

    info = {
        'patients_dic_in_loc': pat_in_loc,
        'patient_count': len(list(patients_dic.keys())),
        'patients_with_epilepsy': len(patients_with_epilepsy),
        'elec_count': elec_count,
        'mean_elec_per_pat': np.mean(elec_count_per_patient),
        'soz_elec_count': soz_elec_count,
        'pse': round(100 * pse, 2),  # percentage of SOZ electrodes
        'pnee': round(100 * non_empty_elec_prop, 2),
        # percentage of non empty elec
        'evt_count': event_count,  # Total count of all types
        'evt_count_per_type': counts,
        'AUC_ROC': auc_roc,
        # Baseline performance in region for these types collapsed
        'conf': conf,  # Sensibility and specificity for the simulator
        'pat_with_x_null': pat_with_x_null,
        'pat_with_y_null': pat_with_y_null,
        'pat_with_z_null': pat_with_z_null,
        'pat_with_loc2_empty': pat_with_loc2_empty,
        'pat_with_loc3_empty': pat_with_loc3_empty,
        'pat_with_loc5_empty': pat_with_loc5_empty,
        'elec_x_null': elec_x_null,
        'elec_y_null': elec_y_null,
        'elec_z_null': elec_z_null,
        'elec_cnt_loc2_empty': elec_cnt_loc2_empty,
        'elec_cnt_loc3_empty': elec_cnt_loc3_empty,
        'elec_cnt_loc5_empty': elec_cnt_loc5_empty,
        'evt_rates': event_rates,  # events per minute for each electrode
        'soz_labels': soz_labels,  # SOZ label of each electrode
        'soz_rates': soz_rates,
        'nsoz_rates': nsoz_rates,
    }
    return info


def first_key(dic):
    return [k for k in dic.keys()][0]
