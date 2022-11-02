from conf import ML_MODELS_TO_RUN, FIG_SAVE_PATH
from db_parsing import EVENT_TYPES, \
    SLEEP_PATIENTS, load_patients, WHOLE_BRAIN_L0C, TWO_KHZ_PATIENTS
from soz_predictor import region_info, first_key
from graphics import plot_global_info_by_loc_table
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)


# Electrodes collection didn't have intraop field, we got all the patients
# from all the events
def get_sleep_patients(electrodes_collection, hfo_collection):
    print('Looking for non intraop patients...')
    elec_cursor = electrodes_collection.find({}, projection=['patient_id'])
    intraop_hfo_cursor = hfo_collection.find(
        {'intraop': '1'},
        projection=['patient_id', 'intraop'])

    sleep_hfo_cursor = hfo_collection.find(
        {'intraop': '0'},
        projection=['patient_id', 'intraop'])

    elec_patients = set()
    for e in elec_cursor:
        elec_patients.add(e['patient_id'])

    intraop_hfo_patients = set()
    for h in intraop_hfo_cursor:
        intraop_hfo_patients.add(h['patient_id'])

    sleep_hfo_patients = set()
    for h in sleep_hfo_cursor:
        sleep_hfo_patients.add(h['patient_id'])

    print('Uncertain patients (appear in intraop and non intraop)')
    in_both = intraop_hfo_patients.intersection(sleep_hfo_patients)
    print(in_both)

    hfo_tot_patients = intraop_hfo_patients.union(sleep_hfo_patients)
    print('Total patient count: {0}'.format(len(elec_patients)))
    print('Patient list: {0}'.format(sorted(list(elec_patients))))

    assert (hfo_tot_patients == elec_patients)

    assert (set(SLEEP_PATIENTS) == sleep_hfo_patients - in_both)
    print('Sleep count: {0}'.format(len(SLEEP_PATIENTS)))
    print('Non intraop Patient list: {0}'.format(SLEEP_PATIENTS))


def global_info_in_locations(elec_collection, evt_collection, intraop=False,
                             locations=None,
                             event_type_names=EVENT_TYPES,
                             restrict_to_tagged_coords=False,
                             restrict_to_tagged_locs=False,
                             saving_path=str(Path(FIG_SAVE_PATH[1], 'table'))):
    print('\nGathering global info...')
    if locations is None:
        locations = {0: [WHOLE_BRAIN_L0C]}
    patients_by_loc = load_patients(elec_collection, evt_collection,
                                    intraop,
                                    loc_granularity=0,
                                    locations=[WHOLE_BRAIN_L0C],
                                    event_type_names=event_type_names,
                                    models_to_run=ML_MODELS_TO_RUN,
                                    load_untagged_coords_from_db=True,
                                    load_untagged_loc_from_db=True,
                                    restrict_to_tagged_coords=restrict_to_tagged_coords,
                                    restrict_to_tagged_locs=restrict_to_tagged_locs)
    data_by_loc = dict()
    whole_brain_name = first_key(patients_by_loc)
    patients_dic = patients_by_loc[whole_brain_name]

    for loc_names in locations.values():
        for loc_name in loc_names:
            loc_info = region_info(patients_dic,
                                   event_types=event_type_names,
                                   location=loc_name if loc_name !=
                                                        whole_brain_name
                                   else None)
            min_pat_count_in_location = 12
            min_pat_with_epilepsy_in_location = 3
            if loc_info['patient_count'] >= min_pat_count_in_location \
                    and loc_info[
                'patients_with_epilepsy'] >= min_pat_with_epilepsy_in_location:

                data_by_loc[loc_name] = dict()
                data_by_loc[loc_name]['patient_count'] = loc_info[
                    'patient_count']
                data_by_loc[loc_name]['patients_with_epilepsy'] = loc_info[
                    'patients_with_epilepsy']
                data_by_loc[loc_name]['elec_count'] = loc_info[
                    'elec_count']
                data_by_loc[loc_name]['soz_elec_count'] = loc_info[
                    'soz_elec_count']
                data_by_loc[loc_name]['PSE'] = loc_info['pse']

                spikes_merge = 0
                for t in event_type_names:
                    if t not in ['Spikes', 'Sharp Spikes']:
                        data_by_loc[loc_name][t + '_N'] = \
                            loc_info['evt_count_per_type'][t]
                    else:
                        spikes_merge += loc_info['evt_count_per_type'][t]
                if 'Spikes' in event_type_names:
                    data_by_loc[loc_name]['Spikes_N'] = spikes_merge

                print('Global info in location: {0} \n {1}'.format(loc_name,
                                                                   data_by_loc[
                                                                       loc_name]))

    plot_global_info_by_loc_table(data_by_loc, saving_path)
