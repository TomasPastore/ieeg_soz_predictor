from statsmodels.stats.multitest import multipletests
from scipy.stats import ks_2samp, mannwhitneyu
from db_parsing import get_granularity, HFO_TYPES
import math as mt
import warnings

warnings.filterwarnings("ignore", module="matplotlib")
warnings.simplefilter(action='ignore', category=FutureWarning)


# 2) HFO rate in SOZ vs NSOZ  ##################################################
# Note: HFO rate is defined in patient.py module as Electrode class method
def gather_stats(stats, feat_name, location, data, types=HFO_TYPES + ['Spikes'],
                 test_names=None):
    '''
    :param stats: global dict to return results
    :param feat_name: feature name to calculate
    :param location: string
    :param data: dictionary with data soz and nsoz by type in location
    :param types: Event types to calculate
    :return: stats global dict updated with stats in location for all the types
    '''
    # Calculating Stats and pvalues for non parametric tests
    if location not in stats.keys():
        stats[location] = dict()
    stats[location][feat_name] = {evt_type: dict() for evt_type in types}
    test_func = {'D': ks_2samp,
                 'U': mannwhitneyu}
    for t in types:
        if min(len(data[t]['soz']), len(data[t]['nsoz'])) == 0:
            print('There is no info for type {t}'.format(t=t))
        else:
            for test_id, test_name in test_names.items():
                stats[location][feat_name][t][test_name] = dict()
                test_f = test_func[test_id]
                S, pval = test_f(data[t]['soz'], data[t]['nsoz'])
                assert (isinstance(pval, float))
                stats[location][feat_name][t][test_name][test_id] = round(S, 2)
                stats[location][feat_name][t][test_name]['pval'] = float(pval)

    return stats


# We have one table per feature example HFO rate
def build_stat_table(locations, feat_name, evt_types, stats, test_names):
    columns = ['Location', 'HFO type']
    test_colors = dict()
    for t_id in test_names.keys():
        columns.append(t_id)
        columns.append('{k}_p-val'.format(k=t_id.lower()))
        test_colors[t_id] = []
    rows = []

    for location in locations:
        for evt_type in evt_types:
            row = [location, evt_type]
            for s_id, s_name in test_names.items():
                try:
                    row.append(
                        stats[location][feat_name][evt_type][s_name][s_id])
                    row.append(
                        stats[location][feat_name][evt_type][s_name]['pval'])
                    test_colors[s_id].append(
                        stats[location][feat_name][evt_type][
                            s_name]['cell_color'])
                except KeyError:
                    row.append('N/A')
                    row.append('N/A')
                    test_colors[s_id].append('red')

            rows.append(row)

    return columns, rows, test_colors


# Stats
def get_soz_nsoz_data(patients_dic,
                      location,
                      feature,
                      types=HFO_TYPES + ['Spikes']):
    '''
    :return: list of dicts[feature][type]= {'soz': v1, 'nsoz': v2}
    '''
    # Structure initialization
    feature_data = []
    fname_sin, fname_cos = 'sin_' + feature, 'cos_' + feature  # only valid
    # for angle
    fnames = [fname_sin, fname_cos] if 'angle' in feature else [
        feature]

    for i, fname in enumerate(fnames):
        feature_data.append({str(fname): dict()})
        for t in types:
            feature_data[i][fname][t] = {'soz': [], 'nsoz': []}
            # print('Initializing data soz nsoz for feature {f} and type {
            # t}'.format(f=fname, t=t))
    granularity = get_granularity(location)
    # Gathering data
    for p in patients_dic.values():
        if location is None or location == 'Whole Brain':
            electrodes = p.electrodes
        else:
            electrodes = [e for e in p.electrodes if
                          location == getattr(e, 'loc{g}'.format(g=
                                                                 granularity))]
        for e in electrodes:
            soz_label = 'soz' if e.soz else 'nsoz'
            for t in types:
                for h in e.events[t]:
                    if 'angle' in feature:
                        if h.info[feature[:-len('_angle')]]:
                            feature_data[0][fname_sin][t][soz_label].append(
                                mt.sin(h.info[feature]))
                            feature_data[1][fname_cos][t][soz_label].append(
                                mt.cos(h.info[feature]))
                    else:
                        feature_data[0][feature][t][soz_label].append(h.info[
                                                                          feature])

    return feature_data


def hfo_types_by_feature(feature):
    if 'spike' in feature:  # spike_angle
        hfo_types = ['RonS', 'Fast RonS']
    elif 'angle' in feature:  # rest of angles
        hfo_types = ['RonO', 'Fast RonO']
    else:  # rest of features
        hfo_types = HFO_TYPES
    return hfo_types


def perform_p_value_correction(stats):
    # build colors
    test_colors = []
    print("Performing multipletest p-value correction...")
    t_pvals = serialize_pvals(stats)
    print("M", len(t_pvals))
    t_reject_flags, t_pvals_adjusted, d3, d4 = multipletests(
        t_pvals, alpha=0.05,
        method='fdr_bh')

    cell_col_by_sig = {
        '*': 'sandybrown',  # pval < 0.01
        '**': 'yellow',  # pval < 0.001
        '***': 'lime'  # pval <  0.0001
    }
    for pval in t_pvals_adjusted:
        if pval < 0.0001:
            cell_col = cell_col_by_sig['***']
        elif pval < 0.001:
            cell_col = cell_col_by_sig['**']
        elif pval < 0.01:
            cell_col = cell_col_by_sig['*']
        else:
            cell_col = 'white'
        test_colors.append(cell_col)

    stats = deserialize_adjusted_pvals(stats, t_pvals_adjusted, test_colors)

    return stats


def serialize_pvals(stats):
    ser_stats = []
    for location, loc_by_feature in stats.items():
        for feature, by_type in loc_by_feature.items():
            for evt_type, by_test_name in by_type.items():
                for test_name in by_test_name.keys():
                    pval = stats[location][feature][evt_type][test_name][
                        'pval']
                    ser_stats.append(pval)
    return ser_stats


def deserialize_adjusted_pvals(stats, t_pvals_adjusted, test_colors):
    i = 0
    for location, loc_by_feature in stats.items():
        for feature, by_type in loc_by_feature.items():
            for type, by_test_name in by_type.items():
                for test_name in by_test_name.keys():
                    stats[location][feature][type][test_name]['pval'] = \
                        t_pvals_adjusted[i]
                    stats[location][feature][type][test_name]['cell_color'] = \
                        test_colors[i]
                    i += 1
    return stats
