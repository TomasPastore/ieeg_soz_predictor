from conf import FIG_SAVE_PATH
from db_parsing import EVENT_TYPES, TYPE_GROUPS, HFO_TYPES, preference_locs
import data_dimensions
from scipy.stats import kruskal
from stats import build_stat_table, gather_stats, get_soz_nsoz_data, \
    hfo_types_by_feature, perform_p_value_correction
from soz_predictor import evt_rate_soz_pred_baseline_localized
from ml_hfo_classifier import ml_hfo_classifier_train, \
    ml_hfo_classifier_validate
import graphics
import utils

import copy
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)


class Driver():

    def __init__(self, elec_collection, evt_collection):
        self.elec_collection = elec_collection
        self.evt_collection = evt_collection

    def run_experiment(self, number, **kwargs):

        NOT_IMPLEMENTED_EXP = NotImplementedError('Experiment not implemented')

        if number == 1:
            print('\nRunning 1) Data dimensions...\n')

            data_dimensions.get_sleep_patients(
                self.elec_collection,
                self.evt_collection)

            Path(FIG_SAVE_PATH[1]).mkdir(parents=True, exist_ok=True)
            saving_path_loc = str(Path(FIG_SAVE_PATH[1], 'dimensions_table'))
            data_dimensions.global_info_in_locations(
                self.elec_collection,
                self.evt_collection,
                intraop=False,
                locations={
                    g: preference_locs(g)
                    for g
                    in [0, 2, 5]},
                event_type_names=EVENT_TYPES,
                restrict_to_tagged_coords=True,
                restrict_to_tagged_locs=True,
                saving_path=saving_path_loc)

        elif number == 2:
            print('\nRunning 2) Data stats analysis...')
            print('Features and HFO rate distributions SOZ vs NSOZ')
            print('Setting parameters for the run...')

            locations = {g: preference_locs(g) for g in [0, 2, 5]}
            location_names = [l for locs in locations.values() for l in locs]

            evt_types_to_load = EVENT_TYPES
            rate_types = copy.deepcopy(evt_types_to_load)
            restrict_to_tagged_coords = True
            restrict_to_tagged_locs = True
            saving_dir = FIG_SAVE_PATH[2]['dir']

            get_histograms = True
            get_stats = True
            get_boxplots = True

            stat_tables_dir = Path(saving_dir, 'stat_tables')
            box_plot_dir = Path(saving_dir, 'violin_plots')  # boxplot
            stat_tables_dir.mkdir(parents=True, exist_ok=True)
            box_plot_dir.mkdir(parents=True, exist_ok=True)
            features = ['HFO_rate', 'duration', 'power_pk',
                        'freq_pk']
            # 'slow_angle', 'delta_angle', 'theta_angle',
            # 'spindle_angle', 'spike_angle'
            stats = dict()  # Saves stats by location for table generation
            test_names = {'D': 'Kolmogorov-Smirnov test', 'U': 'Mann-Whitney '
                                                               'U test'}
            box_plot_data = dict()
            kruskal_data = dict()
            kruskal_features = set()

            pat_dic_filename = str(Path(saving_dir, 'data_by_loc'))
            use_cache = True  # to generate pickle in region_info calculation
            if os.path.exists('{0}.pkl'.format(pat_dic_filename)) and use_cache:
                print('PATH ', '{0}.pkl'.format(pat_dic_filename))
                print('Loading data_by_loc cache')
                data_by_loc = utils.load_object(pat_dic_filename)
            else:
                print('Writing data_by_loc cache')
                data_by_loc = evt_rate_soz_pred_baseline_localized(
                    self.elec_collection,
                    self.evt_collection,
                    intraop=False,
                    load_untagged_coords_from_db=True,
                    # to solve inconsistencies
                    load_untagged_loc_from_db=True,  # to solve inconsistencies
                    restrict_to_tagged_coords=restrict_to_tagged_coords,
                    # In RAM
                    restrict_to_tagged_locs=restrict_to_tagged_locs,  # In RAM
                    evt_types_to_load=evt_types_to_load,
                    evt_types_to_cmp=TYPE_GROUPS,
                    locations=locations,
                    saving_dir=saving_dir,
                    return_pat_dic_by_loc=True,
                    # This is data for later doing ml
                    remove_elec_artifacts=False  # This is false now because we
                    # precomputed the patient list
                )
                utils.save_object(data_by_loc, pat_dic_filename)

            for loc, locs in locations.items():
                for location in locs:
                    saving_hist_dir_feat = str(Path(saving_dir,
                                                    'histograms_soz_nsoz'))

                    saving_hist_dir_feat = str(Path(saving_hist_dir_feat,
                                                    'loc_{g}'.format(g=loc),
                                                    location.replace(' ', '_')))
                    for feature in features:
                        hfo_types = list(set(hfo_types_by_feature(
                            feature)).intersection(evt_types_to_load))
                        if feature == 'HFO_rate':
                            if 'Sharp Spikes' in rate_types:
                                rate_types.remove('Sharp Spikes')
                            datas = [{feature: {t: data_by_loc[location][t +
                                                                         '_rates']
                                                for t in rate_types}}]
                        else:
                            patients_dic = data_by_loc[location]['patients_dic']
                            datas = get_soz_nsoz_data(patients_dic,
                                                      location,
                                                      feature,
                                                      types=hfo_types)  #
                            # Spikes is only for rate
                        for data in datas:
                            feature = utils.first_key(data)
                            data = data[feature]
                            for evt_type, rates in data.items():
                                for z in ['soz', 'nsoz']:
                                    if feature not in kruskal_data.keys():
                                        kruskal_features.add(feature)
                                        kruskal_data[feature] = {
                                            evt_type: dict(soz=[], nsoz=[]) for
                                            evt_type
                                            in HFO_TYPES + [
                                                'Spikes']}
                                    kruskal_data[feature][evt_type][z].append(
                                        rates[z])
                            if feature not in box_plot_data.keys():
                                box_plot_data[feature] = dict()  # for each loc
                            box_plot_data[feature][
                                location] = data  # dict by ty

                            # Generates SOZ vs NSOZ histograms
                            if get_histograms:
                                graphics.plot_types_feature_distribution(
                                    data=data,
                                    feature=feature,
                                    location=location,
                                    hfo_types=hfo_types,
                                    saving_dir=saving_hist_dir_feat)

                            # Calculate stats
                            # returns data for generating the statistic tables
                            if get_stats:
                                stats = gather_stats(
                                    stats,
                                    feat_name=feature,
                                    location=location,
                                    data=data,
                                    types=hfo_types if feature != 'HFO_rate'
                                    else
                                    rate_types,
                                    test_names=test_names)
            kruskal_columns = ['Property', 'HFO type', 'Label', 'H', 'p-val']
            kruskal_rows = []
            for f in kruskal_features:
                for t, data in kruskal_data[f].items():
                    for z in ['soz', 'nsoz']:
                        if len(data[z]) > 0:
                            H, pval = kruskal(*data[z])  # NON PARAMETRIC NOW
                            label = 'SOZ' if z == 'soz' else 'NSOZ'
                            kruskal_rows.append([graphics.pretty_print(f),
                                                 t, label, round(H, 4),
                                                 format(pval, '.2e')])

            graphics.kruskal_table(kruskal_columns, kruskal_rows,
                                   saving_path=str(Path(stat_tables_dir,
                                                        'kruskal_results'))
                                   )

            stats = perform_p_value_correction(stats)

            for feature in features:
                hfo_types = list(set(hfo_types_by_feature(
                    feature)).intersection(evt_types_to_load))
                fname_sin, fname_cos = 'sin_' + feature, 'cos_' + feature
                fnames = [fname_sin, fname_cos] if 'angle' in feature else [
                    feature]
                for fname in fnames:
                    # Boxplot
                    # Grouped boxplot
                    if get_boxplots:
                        graphics.plot_feature_box_plot(box_plot_data[fname],
                                                       # By region
                                                       fname,
                                                       box_plot_dir)

                    # Stats
                    if get_stats:
                        columns, rows, test_colors = build_stat_table(
                            locations=location_names,
                            feat_name=fname,
                            evt_types=hfo_types if fname != 'HFO_rate' else
                            rate_types,
                            stats=stats,
                            test_names=test_names)

                        graphics.pval_table(columns, rows,
                                            saving_path=str(Path(
                                                stat_tables_dir,
                                                '{f}_table'.format(
                                                    f=fname))),
                                            test_names=test_names,
                                            test_colors=test_colors
                                            )

        elif number == 3:
            print('\nRunning exp 3) Predicting SOZ with baseline rates...')
            print('Setting parameters for the run...')
            evt_types_to_load = EVENT_TYPES
            evt_types_to_cmp = TYPE_GROUPS
            subtypes_to_load = None  # 'delta', 'theta', 'slow', 'spindle'
            restrict_to_tagged_coords = True
            restrict_to_tagged_locs = True
            locations = {g: preference_locs(g) for g in [0, 2, 5]}
            saving_dir = FIG_SAVE_PATH[3]['dir']
            evt_rate_soz_pred_baseline_localized(
                self.elec_collection,
                self.evt_collection,
                intraop=False,
                load_untagged_coords_from_db=True,
                load_untagged_loc_from_db=True,
                restrict_to_tagged_coords=restrict_to_tagged_coords,
                restrict_to_tagged_locs=restrict_to_tagged_locs,
                evt_types_to_load=evt_types_to_load,
                subtypes_to_load=subtypes_to_load,
                evt_types_to_cmp=evt_types_to_cmp,
                locations=locations,
                saving_dir=saving_dir,
                plot_rocs=True)
        elif number == 4:
            print('\nRunning exp 4) Machine learning HFO classifiers...')
            print('Setting parameters for the run')
            subtypes_to_load = None  # None corresponds to all subtypes
            evt_types_to_load = ['Fast RonO']  # ['Fast RonO']# 'Rons'
            location = 'Whole Brain'
            locations = {0: [location]}

            # kwargs conf for ml has priority than variables above (defaults)
            print('KWARGS ', kwargs)
            evt_types_to_load = kwargs.pop('evt_types_to_load', evt_types_to_load)
            locations = kwargs.pop('locations', locations)

            restrict_to_tagged_coords = True
            restrict_to_tagged_locs = True
            model_name = 'SGD'
            use_coords = False
            saving_dir = FIG_SAVE_PATH[4]['dir']

            data_by_loc = evt_rate_soz_pred_baseline_localized(
                self.elec_collection,
                self.evt_collection,
                intraop=False,
                load_untagged_coords_from_db=True,
                load_untagged_loc_from_db=True,
                restrict_to_tagged_coords=restrict_to_tagged_coords,
                restrict_to_tagged_locs=restrict_to_tagged_locs,
                evt_types_to_load=evt_types_to_load,
                subtypes_to_load=subtypes_to_load,
                evt_types_to_cmp=[[t] for t in evt_types_to_load],
                locations=locations,
                saving_dir=saving_dir,
                return_pat_dic_by_loc=True,
                plot_rocs=False,
                remove_elec_artifacts=False)

            for locs in locations.values():
                for location in locs:
                    patients_dic = data_by_loc[location]['patients_dic']
                    for hfo_type in evt_types_to_load:
                        # Here below the location parameter is just for passing
                        # the string since the data dictionary has already
                        # been filtered
                        validation_patients, ml_data = ml_hfo_classifier_train(
                            patients_dic,
                            location=location,
                            hfo_type=hfo_type,
                            use_coords=use_coords,
                            model_name=model_name,
                            saving_dir=saving_dir)

                        ml_hfo_classifier_validate(hfo_type, location,
                                                   validation_patients,
                                                   ml_data, saving_dir)

        else:
            raise NOT_IMPLEMENTED_EXP
