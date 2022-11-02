import copy
# Reviewed
from sklearn.metrics import roc_curve


def Hippocampal_RonO_gradual_filters(elec_collection, evt_collection):
    model_name = 'XGBoost'
    models_to_run = [model_name]
    tol_fprs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    compare_baseline_vs_ml(elec_collection, evt_collection, 'Hippocampus',
                           'RonS',
                           tol_fprs=tol_fprs, models_to_run=models_to_run,
                           comp_with='{0} '.format(model_name))


def compare_baseline_vs_ml(patients_dic,  # Patients that built the baseline
                           plot_data_by_loc,  # has baseline ROC info
                           location,  # loc name
                           hfo_type,  # hfo type name
                           use_coords,
                           target_patients_id='MODEL_PATIENTS',
                           ml_models=['XGBoost'],
                           tol_fprs=[0.6],  # HFO filter thresh to discard
                           # fisiological hfos below the proba thresh associate
                           sim_recall=None,
                           saving_path=None):
    target_patients = ml_hfo_classifier(patients_dic, location, hfo_type,
                                        use_coords, target_patients_id,
                                        ml_models, sim_recall, saving_path)

    for model_name in ml_models:

        simulating = sim_recall is not None
        if simulating and model_name != 'Simulator':
            # this is cause we need to run at least one other mode  l to
            # simulate,
            # but then we just plot the simulated
            continue

        # Maps predictions and probas to list in linear search of target pat
        labels, preds, probs = gather_folds(model_name, hfo_type,
                                            target_patients, estimator=np.mean)

        print(
            'Displaying metrics for {t} in {l} ml HFO classifier using {'
            'm}'.format(
                t=hfo_type, l=location, m=model_name))
        print_metrics(model_name, hfo_type, labels, preds, probs)

        # SOZ HFO RATE MODEL
        fpr, tpr, thresholds = roc_curve(labels, probs)
        for tol_fpr in tol_fprs:
            thresh = get_soz_confidence_thresh(fpr, tpr, thresholds,
                                               tolerated_fpr=tol_fpr)
            filtered_pat_dic = phfo_thresh_filter(target_patients,
                                                  hfo_type,
                                                  thresh=thresh,
                                                  model_name=model_name)

            reg_loc = location if location != 'Whole Brain' else None
            loc_info = region_info(filtered_pat_dic,
                                   event_types=[hfo_type],
                                   flush=True,  # el flush es importante
                                   # porque hay que actualizar los counts
                                   conf=sim_recall,
                                   location=reg_loc,
                                   )  # calcula la info para la roc con la
            # prob asociada al fpr tolerado
            fig_model_name = '{t}_{m}_fpr_{f}'.format(t=hfo_type,
                                                      m=model_name,
                                                      f=tol_fpr)
            plot_data_by_loc[location][fig_model_name] = loc_info

    comp_with = ''  # DONT remember why was this, i think that for colors in
    # simu
    graphics.event_rate_by_loc(plot_data_by_loc,
                               metrics=['pse', 'pnee', 'auc'],
                               title='HFO rate baseline VS ML pHFO filters: {'
                                     't} in {l}'.format(
                                   t=hfo_type, l=location),
                               roc_saving_path=str(Path(saving_path,
                                                        location, 'roc')),
                               colors='random' if comp_with == '' else None,
                               conf=sim_recall)


def get_soz_confidence_thresh(fpr, tpr, thresholds, tolerated_fpr):
    def print_thresh_info(tol_fpr, i):
        print('For tolerated FPR {t_fpr} --> Proba_thresh: {p_thresh}, '
              'TPR: {tpr}'.format(t_fpr=tol_fpr, p_thresh=thresholds[i],
                                  tpr=tpr[i]))

    for i in range(len(fpr)):
        if fpr[i] == tolerated_fpr:
            print_thresh_info(tolerated_fpr, i)
            return thresholds[i]
        elif fpr[i] < tolerated_fpr:
            continue
        elif fpr[i] > tolerated_fpr:
            if abs(fpr[i] - tolerated_fpr) <= abs(fpr[i - 1] - tolerated_fpr):
                print_thresh_info(tolerated_fpr, i)
                return thresholds[i]
            else:
                print_thresh_info(tolerated_fpr, i - 1)
                return thresholds[i - 1]


# Reviewed
# Filters the Events whose predicted probability of being SOZ is greater than
# the threshold given by parameter
def phfo_thresh_filter(target_patients, hfo_type_name, thresh=None,
                       model_name='XGBoost'):
    filtered_pat_dic = dict()
    for p in target_patients:
        p_copy = copy.deepcopy(p)
        for e_copy, e in zip(p_copy.electrodes, p.electrodes):
            e_copy.events[hfo_type_name] = []
            for h in e.events[hfo_type_name]:
                if h.info['proba'] >= thresh:
                    e_copy.add(event=copy.deepcopy(h))
            e_copy.flush_cache([hfo_type_name])  # recalculates event counts
        filtered_pat_dic[p.id] = p_copy

    return filtered_pat_dic
