import numpy as np
from random import choices


def rm_stop_cond(physio_cnts, artifact_cnts, p_name, art_ratio_mean,
                 physio_mean):
    cond_ratio = (physio_cnts[p_name] == 0 and artifact_cnts[p_name] > 0) \
                 or (physio_cnts[p_name] != 0 and
                     ((artifact_cnts[p_name] / physio_cnts[
                         p_name]) > art_ratio_mean))
    cond_less_art_than_physio_mean = artifact_cnts[p_name] > physio_mean
    return cond_less_art_than_physio_mean


def artifact_filter(hfo_type, patients_dic):
    '''
    Filters Fast RonO near 300 HZ and RonO near 180 Hz electrical artifacts.
    :param hfo_type: The hfo type with electrical artifacts
    :param patients_dic: Patient, Electrode, Event data structures
    :return: modified patients dic for type hfo_type
    '''
    print('\nEntering filter for electrical artifacts')
    remove_from_elec_by_pat = {p_name: [] for p_name in patients_dic.keys()}
    # For each patient I keep a list of elec names where we can gradually
    # remove candidates and update
    if hfo_type == 'Fast RonO':
        artifact_freq = 300  # HZ
        art_radius = 20  # HZ
        pw_line_int = 60  # HZ

    elif hfo_type == 'RonO':
        artifact_freq = 180  # HZ
        art_radius = 20  # HZ
        pw_line_int = -60  # HZ # Makes 120 HZ the physiological band to look
    else:
        print('Not implemented filter type')
        raise NotImplementedError()

    artifact_cnts = dict()  # art_freq +- art_radius HZ event counts per pat
    physio_cnts = dict()  # physio_freq +- art_radius HZ event counts per pat
    artifact_ratios = []  # art_freq/physio_freq per patient
    for p_name, p in patients_dic.items():
        artifact_cnt_tmp = 0
        physio_cnt_tmp = 0
        for e in p.electrodes:
            for evt in e.events[hfo_type]:
                if (artifact_freq - art_radius) <= evt.info['freq_av'] and \
                        evt.info['freq_av'] <= (artifact_freq + \
                                                art_radius):
                    artifact_cnt_tmp += 1
                    remove_from_elec_by_pat[p_name].append(e.name)

                elif (artifact_freq + pw_line_int - art_radius) <= evt.info[
                    'freq_av'] and \
                        evt.info['freq_av'] <= (artifact_freq +
                                                pw_line_int + art_radius):
                    physio_cnt_tmp += 1
        artifact_cnts[p_name] = artifact_cnt_tmp
        physio_cnts[p_name] = physio_cnt_tmp
        artifact_ratios.append(
            artifact_cnt_tmp / physio_cnt_tmp if physio_cnt_tmp > 0
            else 1)

    # Saving stats
    art_ratio_mean = np.mean(artifact_ratios)
    # art_ratio_mean = np.median(artifact_ratios)
    artifact_ratio_std = np.std(artifact_ratios, ddof=1)

    physio_mean = np.mean(list(physio_cnts.values()))
    # physio_mean = np.median(list(physio_cnts.values()))
    physio_std = np.std(list(physio_cnts.values()), ddof=1)

    artifact_mean = np.mean(list(artifact_cnts.values()))
    # artifact_mean = np.median(list(artifact_cnts.values()))
    artifact_std = np.std(list(artifact_cnts.values()), ddof=1)

    print('-----------------------------------')
    print('\n{h} artifact ratios ({art}+-{a})/({p}+-{a}) HZ'.format(
        h=hfo_type, art=artifact_freq, a=art_radius,
        p=artifact_freq + pw_line_int))
    print('\tSample artifact ratios mean', art_ratio_mean)
    print('\tSample artifact ratios std', artifact_ratio_std)

    print('\n{h} Physiological ({p} HZ +- {a})'.format(h=hfo_type,
                                                       p=artifact_freq +
                                                         pw_line_int,
                                                       a=art_radius))
    print('\tSample physiological mean', physio_mean)
    print('\tSample physiological std', physio_std)

    print('\n{h} Artifacts ({art} HZ +- {a})'.format(h=hfo_type,
                                                     art=artifact_freq,
                                                     a=art_radius))
    print('\tSample artifact mean', artifact_mean)
    print('\tSample artifact std', artifact_std)
    print('-----------------------------------')

    # Removing artifacts
    for p_name, p in patients_dic.items():
        remove_cnt = 0
        while (rm_stop_cond(physio_cnts, artifact_cnts, p_name,
                            art_ratio_mean, physio_mean)):
            # Choose an electrode with artifacts randomly weighted by
            # artifact count by electrode
            elec_to_rmv = choices(remove_from_elec_by_pat[p_name], k=1)[0]
            remove_from_elec_by_pat[p_name].remove(elec_to_rmv)  # update cnt
            artifact_cnts[p_name] -= 1  # update artifact cnt
            electrode = p.get_electrode(elec_to_rmv)
            # Removes one event randomly from the artifacts in the
            # channel (300 +- art_radius)
            electrode.remove_rand_evt(hfo_type=hfo_type,
                                      art_radius=art_radius)
            remove_cnt += 1

        for e in p.electrodes:
            e.flush_cache([hfo_type])  # Recalc events counts for hfo rate

        print('For patient {0} we remove {1} events'.format(p_name,
                                                            remove_cnt))

    return patients_dic
