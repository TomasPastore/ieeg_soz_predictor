from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from functools import partial
from conf import FIG_FOLDER_DIR, ORCA_EXECUTABLE, FIG_SAVE_PATH
from db_parsing import get_granularity, HFO_TYPES, WHOLE_BRAIN_L0C

# mplstyle.use(['ggplot',1 'fast'])
color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


# Reviewed
# 1) Global info by location table
def plot_global_info_by_loc_table(data_by_loc, saving_path):
    np.random.seed(1)
    sorted_types = sorted(HFO_TYPES + ['Spikes'])
    col_colors = []
    rows = []
    for loc, data in data_by_loc.items():
        row = []
        granularity = get_granularity(loc)
        row.append(granularity)
        row.append(loc)
        row.append(data['patient_count'])
        row.append(data['patients_with_epilepsy'])
        row.append(data['elec_count'])
        # row.append(data['soz_elec_count'])
        row.append(data['PSE'])
        for evt_type in sorted_types:
            row.append(data[evt_type + '_N'])
        rows.append(tuple(row))

    # Order by granularity, loc_name
    rows = sorted(rows, key=lambda x: (x[0], x[1]))

    for row in rows:
        col_colors.append(color_by_gran(row[0]))
    col_names = ['Location', '#Patients', '#SOZ Patients', '#Elec',
                 'PSE'] + ['#{t}'.format(t=t) for t in sorted_types]  # PSE
    col_width = [len(col_name) for col_name in col_names]
    for r in range(len(rows)):
        for c in range(1, len(rows[0])):  # granularity out
            if len(str(rows[r][c])) > col_width[c - 1]:
                col_width[c - 1] = len(str(rows[r][c]))

    print('\n Generating table...')
    # print('Columns {0}'.format(col_names))
    # print('Column widths: {0}'.format(col_width))
    col_width = [20, 10, 10, 6, 6, 7, 7, 7, 7, 7]
    col_values = ['{b}{c}{b}'.format(b='<b>', c=col_names[i].ljust(
        col_width[i])) for i in range(len(col_names))]
    columns = [[r[i + 1] for r in rows] for i in range(len(col_names))]
    fig = go.Figure(
        data=[go.Table(
            columnwidth=[col_width[i] for i in range(len(col_width))],
            header=dict(
                values=col_values,
                line_color='black', fill_color='white',
                align='left', font=dict(color='black', size=12)
            ),
            cells=dict(
                values=columns,
                # toma columnas, el +1 es porque no estoy imprimiendo granularity
                fill_color=[np.array(col_colors) for i in
                            range(len(col_names))],
                align='left', font=dict(color='black', size=12)
            ))
        ])

    row_height = 45
    header_h = 40
    table_h = header_h + row_height * len(rows)

    fig.update_layout(
        autosize=False,
        height=table_h,
        width=80 * len(col_names),
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=10,
            pad=2
        ))
    orca_save(fig, saving_path, scale=1)  # .2
    fig.show()

    # Saves table as csv
    row_values = [[r[i + 1] for i in range(len(col_names))] for r in rows]
    colu_names = [col_names[i].ljust(col_width[i]) for i in range(len(col_names))]
    table_df = pd.DataFrame(data=np.array(row_values), columns=colu_names)
    table_df.to_csv('{path}.{fmt}'.format(path=saving_path, fmt='csv'))
    table_df.to_excel('{path}.{fmt}'.format(path=saving_path, fmt='xlsx'))


# Reviewed
# 2) HFO rate comparison and features comparison

# Histograms
def plot_types_feature_distribution(data, feature, location, hfo_types,
                                    saving_dir):
    saving_dir = str(Path(saving_dir, feature))
    Path(saving_dir).mkdir(parents=True, exist_ok=True)
    fig_path = str(Path(saving_dir, feature + '_distrs.pdf'))
    # print('Distributions saving path: {0}'.format(fig_path))
    sns.set_style("white")
    # Plot
    subplot_index = {'RonO': 1, 'RonS': 2, 'Fast RonO': 3, 'Fast RonS': 4}
    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

    fig = plt.figure(figsize=(10, 7), dpi=80)
    fig.suptitle('{l} {feat} SOZ vs NSOZ distributions'.format(
        l=location,
        feat=pretty_print(feature)),
        fontsize=20)
    for evt_type in hfo_types:
        axe = plt.subplot(
            '{r}{c}{i}'.format(r=2, c=2, i=subplot_index[evt_type]))
        axe.set_title(evt_type, fontdict={'fontsize': 14}, loc='left')
        axe.set_xlabel(feature.capitalize(), fontdict={'fontsize': 12})
        axe.set_ylabel('Frequency', fontdict={'fontsize': 12})
        soz_data = data[evt_type]['soz']
        nsoz_data = data[evt_type]['nsoz']
        '''
        sns.distplot(soz_data, color="red",
                     label="SOZ", **kwargs)
        sns.distplot(nsoz_data, color="green",
                     label="NSOZ", **kwargs)
        '''
        soz_pd = pd.DataFrame(data={pretty_print(feature): soz_data,
                                    'Elec. label': ['SOZ'] * len(soz_data)})
        nsoz_pd = pd.DataFrame(data={pretty_print(feature): nsoz_data,
                                     'Elec. label': ['Non-SOZ'] * len(nsoz_data)})
        data_pd = pd.concat([soz_pd, nsoz_pd])
        log_scale = True if feature == 'HFO_rate' else False
        if log_scale:
            replace_with = 1
            for e in soz_data + nsoz_data:
                if 0 < e < replace_with:
                    replace_with = e
            # print(replace_with) #first min greater than 0

            # Option 1 print
            # data_pd[pretty_print(feature)] = data_pd[pretty_print(
            #    feature)].replace(0, replace_with ) #take the min
            # after 0, log of 0 is not defined
            elec_count = len(data_pd[pretty_print(feature)])
            cero_rates = len(data_pd[data_pd[pretty_print(
                feature)] == 0])
            data_pd = data_pd[data_pd[pretty_print(feature)] > 0]
            assert (len(data_pd[pretty_print(feature)]) == (elec_count - cero_rates))

            non_cero_elec = round((elec_count - cero_rates) / elec_count, 2)
            hrate = ' \nElec_non_zero_rate {p}'.format(p=non_cero_elec)
        else:
            hrate = ''
        axe = sns.histplot(data=data_pd, x=pretty_print(feature),
                           hue='Elec. label', stat='probability',
                           element="step",
                           common_bins=True, common_norm=False, kde=True,
                           thresh=0,
                           pthresh=None,
                           log_scale=log_scale)

        # cuanto significa el valor en xlim index comparado con el resto
        def pval_rate_higher(data, xlim_index):
            return len([r for r in data if r >= data[xlim_index]]) / len(data)

        def pval_rate_lower(data, xlim_index):
            return len([r for r in data if r <= data[xlim_index]]) / len(data)

        def get_xlim_higher(data, pval=0.005):
            data = sorted(data)
            xlim_index = len(data) - 1
            while (pval_rate_higher(data,
                                    xlim_index) <= pval):  # si no tiene tantas
                # obs no lo ploteo para mejorar la visualizacion con zoom.
                xlim_index -= 1
            return data[xlim_index]

        def get_xlim_lower(_data, pval=0.005):
            _data = sorted(_data)
            xlim_index = 1
            while pval_rate_lower(_data, xlim_index) <= pval:  # si no tiene
                # tantas
                # obs no lo ploteo para mejorar la visualizacion con zoom.
                xlim_index += 1
            return _data[xlim_index]

        xlim_low = min(get_xlim_lower(soz_data), get_xlim_lower(nsoz_data))
        xlim_high = max(get_xlim_higher(soz_data), get_xlim_higher(nsoz_data))
        # axe.set_xlim([xlim_low, xlim_high])

        soz_mean = round(np.mean(data[evt_type]['soz']), 2)
        nsoz_mean = round(np.mean(data[evt_type]['nsoz']), 2)

        soz_median = round(np.median(data[evt_type]['soz']), 2)
        nsoz_median = round(np.median(data[evt_type]['nsoz']), 2)
        # For text block coords (0, 0) is bottom and (1, 1) is top
        axe.text(x=0.55 if not log_scale else 0.45,  # upper right
                 y=0.75,
                 s='SOZ mean: {soz_mean}\n'
                   'SOZ median: {soz_med}\n'
                   'nSOZ mean: {nsoz_mean}\n'
                   'nSOZ median: {nsoz_med}\n'
                   '----------------\n'
                   'Mean dif {mean_dif} \n'
                   'Median dif {med_dif} \n'
                   '+ rate elec. {hrate}'.format(
                     soz_mean=round(soz_mean, 2),
                     soz_med=round(soz_median, 2),
                     nsoz_mean=round(nsoz_mean, 2),
                     nsoz_med=round(nsoz_median, 2),
                     mean_dif=round(soz_mean - nsoz_mean, 2),
                     med_dif=round(soz_median - nsoz_median, 2),
                     hrate=hrate),
                 bbox=dict(facecolor='grey', alpha=0.5),
                 transform=axe.transAxes, fontsize=7)
        # axe.legend(l, loc='center right', prop={'size': 8})

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    fig.savefig(fig_path)
    plt.close(fig)
    # plt.show()


def plot_feature_box_plot(box_plot_data, feature, saving_dir):
    '''
    :param box_plot_data: has the data by location, type, nsoz soz for the feat
    :param feature: string
    :param saving_dir: path
    :return:
    '''
    box_plot_df_by_type = dict()
    event_types = []
    mins_by_type = dict()
    max_by_type = dict()
    for location, data_by_type in box_plot_data.items():
        for evt_type, data_soz_nsoz in data_by_type.items():
            if evt_type not in box_plot_df_by_type.keys():
                box_plot_df_by_type[evt_type] = []
                event_types.append(evt_type)
            if feature in ['HFO_rate']:
                all_val = data_soz_nsoz['soz'] + data_soz_nsoz['nsoz']
                minimo = 1
                maximo = 0
                for e in all_val:
                    assert e >= 0
                    if e > 0 and e < minimo:
                        minimo = e
                    if e > 0 and e > maximo:
                        maximo = e
                if evt_type in mins_by_type.keys():
                    mins_by_type[evt_type] = min(minimo, mins_by_type[evt_type])
                    max_by_type[evt_type] = max(maximo, max_by_type[evt_type])
                else:
                    mins_by_type[evt_type] = minimo
                    max_by_type[evt_type] = maximo
            for k in ['soz', 'nsoz']:
                data = data_soz_nsoz[k]
                d = {pretty_print(feature): data,
                     'Elec. label': ['SOZ' if k == 'soz' else 'Non-SOZ'] * len(
                         data),
                     'Brain Region': [rem_lobe(location)] * len(data)}
                box_plot_df_by_type[evt_type].append(pd.DataFrame(data=d))
    for evt_type in event_types:
        fig = plt.figure(10, figsize=(14, 4))
        title = '{t} {f} SOZ vs Non-SOZ'.format(
            t=evt_type,
            f=pretty_print(feature))
        fig.suptitle(title, fontsize=18)

        data_pd = pd.concat(box_plot_df_by_type[evt_type])
        if feature in ['HFO_rate']:
            data_pd[pretty_print(feature)] = data_pd[pretty_print(
                feature)].replace(0, mins_by_type[evt_type])
            # This is to map log10 to rate feature, currently I'm using y log
            # scale in the figure layout
            data_pd[pretty_print(feature)] = list(map(np.log10, data_pd[
                pretty_print(feature)]))
        '''
        # Violin plot with plotly
        df = data_pd
        fig = go.Figure()
        pointpos_nsoz = [-0.5] * len(pd.unique(df['Brain Region']))
        pointpos_soz = [0.5, 0.5, 0.5, 0.5] * len(pd.unique(df['Brain '
                                                                 'Region']))
        show_legend = [False] * len(pd.unique(df['Brain Region']))
        show_legend[0] = True
        
        for i in range(0, len(pd.unique(df['Brain Region']))):
            y = df[pretty_print(feature)][(df['Elec. label'] ==
                                                           'Non-SOZ') &
                                                       (df['Brain Region'] ==
                                                        pd.unique(df['Brain 
                                                        Region'])[
                                                            i])]
            fig.add_trace(go.Violin(x=df['Brain Region'][(df['Elec. label'] ==
                                                          'Non-SOZ') &
                                                (df['Brain Region'] ==
                                                 pd.unique(df['Brain 
                                                 Region'])[i])],
                                    y=df[pretty_print(feature)][(df['Elec. 
                                    label'] ==
                                                           'Non-SOZ') &
                                                       (df['Brain Region'] ==
                                                        pd.unique(df['Brain 
                                                        Region'])[
                                                            i])],
                                    legendgroup='Non-SOZ', 
                                    scalegroup='Non-SOZ', name='Non-SOZ',
                                    side='negative',
                                    line_color='green',
                                    box=dict(visible=True, width=0.8),
                                    showlegend=show_legend[i])
                          )
            fig.add_trace(go.Violin(x=df['Brain Region'][(df['Elec. label'] ==
                                                          'SOZ') &
                                                (df['Brain Region'] ==
                                                 pd.unique(df['Brain 
                                                 Region'])[i])],
                                    y=df[pretty_print(feature)][(df['Elec. '
                                                                    'label']
                                                                 == 'SOZ') &
                                                       (df['Brain Region'] ==
                                                        pd.unique(df['Brain 
                                                        Region'])[
                                                            i])],
                                    legendgroup='SOZ', scalegroup='SOZ',
                                    name='SOZ',
                                    side='positive',
                                    line_color='red',
                                    box=dict(visible=True, width=0.5),
                                    showlegend=show_legend[i])
                          )

        # update characteristics shared by all traces
        fig.update_traces(meanline_visible=True,
                          points='outliers',  # show all points
                          jitter=1,#0.05,
                          # add some jitter on points for better visibility
                          scalemode='count')  # scale violin plot area with 
                          total count
        if feature == 'HFO_rate':
            pass
            #fig.update_yaxes(nticks=20)
            #fig.update_yaxes(type='log')#autorange=True
            #fig.update_layout(
            #    yaxis=dict(type='log')
            #)
        fig.update_xaxes(title_text='Brain Region', title_font = {
            "size": 14})
        fig.update_yaxes(title_text=pretty_print(feature),  title_font = {
            "size": 14})
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            violingap=0, violingroupgap=0
            )
        saving_path = str(Path(saving_dir, evt_type.replace(' ', '_'), feature,
                               evt_type.replace(' ', '_') + '_' + feature))
        Path(saving_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(file=saving_path+'_plotly.html')
        fig.write_image(file=saving_path+'_plotly.pdf')
        fig.write_image(file=saving_path+'_plotly.eps')
        #fig.show()
        '''

        # Seaborn
        if feature == 'HFO_rate':
            # ax=plt.gca(yscale="log") #base default is 10
            # print('MAX RATE {0} {1}'.format(evt_type, max_by_type[evt_type]))
            # ni sacando los 0 puedo plotear en logscale
            # data_pd = data_pd[data_pd[pretty_print(feature)] > 0]
            ax = None
        else:
            ax = None
        ax = sns.violinplot(
            x='Brain Region',
            y=pretty_print(feature),
            hue='Elec. label',
            data=data_pd,
            palette="Set1",
            # split=True,
            inner='box',
            ax=ax,
            scale='count'
        )
        '''
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)
        ax.set_ylabel(pretty_print(feature), fontdict={'fontsize': 18})
        sns.set_context("paper", rc={"font.size": 18, "axes.titlesize": 18,
                                     "axes.labelsize": 18})
        ax.legend(prop=dict(size=16))

        if feature == 'HFO_rate':
            #ax.set(ylim = (mins_by_type[evt_type] * 0.1, max_by_type[
            # evt_type] * 10))
                    
        '''

        saving_path = str(Path(saving_dir, evt_type.replace(' ', '_'), feature,
                               evt_type.replace(' ', '_') + '_' + feature))
        Path(saving_path).parent.mkdir(parents=True, exist_ok=True)
        for fmt in ['pdf', 'png', 'eps']:
            if fmt == 'eps':
                print('Eps warning')
            fig.savefig('{dir}.{fmt}'.format(dir=saving_path, fmt=fmt),
                        bbox_inches='tight')
            if fmt == 'pdf':
                # plt.show()
                pass
            plt.close(fig)


def pval_table(columns, rows, saving_path=None, test_names=None,
               test_colors=None):
    for r in rows:
        for col_index in list(
                reversed(sorted([columns.index(name) + 1 for name in
                                 test_names.keys()]))):
            test_name = columns[col_index - 1]
            print('appending color for index ', col_index)
            r.append(test_colors[test_name].pop(0))
            if isinstance(r[col_index], float):
                r[col_index] = format(r[col_index], '.2e')
    rows = sorted(rows, key=lambda x: x[1])  # Order by 1 HFO type name, 0 loc
    white_col = ['white'] * len(rows)

    def color_col_by_index(i):
        if test_colors is not None:
            if i == 3:
                colors = []
                for r in rows:
                    print('removing color for i ', 3)
                    colors.append(r.pop(-1))
                return colors
            elif i == 5:
                print('removing color for i ', 5)

                colors = []
                for r in rows:
                    colors.append(r.pop(-1))
                return colors
            else:
                return white_col
        else:
            raise RuntimeError("Test_colors should not be None")

    fill_color = [color_col_by_index(i) for i in range(len(columns))]
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=['<b>' + c + '</b>' for c in columns],
                line_color='black', fill_color='white',
                align='left', font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[[r[i] for r in rows] for i in range(len(columns))],
                line_color='black', fill_color=fill_color,
                align='right', font=dict(color='black', size=12)
            ))
        ])
    col_width = 100
    row_height = 50
    fig.update_layout(
        autosize=False,
        width=col_width * len(columns),
        height=row_height * (len(rows) + 1),
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=3
        ))
    orca_save(fig, saving_path)
    # fig.show()


def kruskal_table(columns, rows, saving_path=None):
    rows = sorted(rows, key=lambda x: (x[0], x[1]))  # Order by feature and HFO
    # type name

    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=['<b>' + c + '</b>' for c in columns],
                line_color='black', fill_color='white',
                align='left', font=dict(color='black', size=14)
            ),
            cells=dict(
                values=[[r[i] for r in rows] for i in range(len(columns))],
                line_color='black', fill_color='white',
                align='right', font=dict(color='black', size=12)
            ))
        ])
    col_width = 100
    row_height = 50
    fig.update_layout(
        autosize=False,
        width=col_width * len(columns),
        height=row_height * (len(rows) + 1),
        margin=dict(
            l=20,
            r=20,
            b=20,
            t=20,
            pad=3
        ))
    orca_save(fig, saving_path)
    # fig.show()


# 3) Predicting SOZ with rate, used in almost all the steps to plot ROCs
# Reviewed
# Plots ROCs for SOZ predictor by hfo rate for different locations and event
# types
def short_name(hfo_name):
    return 'F' + hfo_name.split(' ')[1] if hfo_name.startswith('Fast') else hfo_name


def event_rate_by_loc(hfo_type_data_by_loc, zoomed_type=None,
                      metrics=['pse', 'pnee', 'auc'],
                      title=None, colors=None, conf=None,
                      roc_saving_path=FIG_FOLDER_DIR + 'fig',
                      change_tab_path=None, partial_roc=True):
    print('Plotting event rate by loc...')
    # plt.ioff() #enable show
    Path(roc_saving_path).parent.mkdir(parents=True, exist_ok=True)
    # Subplots frames
    subplot_count = len(hfo_type_data_by_loc.keys())
    tile_horizontal = False
    if subplot_count == 1:
        fig = plt.figure(42, figsize=(6, 6))  # 6 6 whole brain
        #fig = plt.figure(42, figsize=(4, 5)) For frontal lobe
    elif subplot_count == 2:
        fig = plt.figure(42, figsize=(8, 5))
    elif subplot_count > 2:
        if tile_horizontal:
            fig = plt.figure(42, figsize=(16, 12))  #  width, height (16, 12)
        else:
            fig = plt.figure(42, figsize=(6, 8))  #
    else:
        raise RuntimeError('Subplot count not implemented')

    if tile_horizontal:
        rows = 1
        cols = subplot_count  # Tile Horizontally
    else:
        if subplot_count == 1:
            rows = 1
            cols = 1
        elif subplot_count <= 2:
            rows = 1
            cols = 2
        elif subplot_count <= 4:
            rows = 2
            cols = 2
        elif subplot_count <= 6:
            rows = 2
            cols = 3
        elif subplot_count < 10:
            rows = 3
            cols = 3
        else:
            raise RuntimeError('Subplot count not implemented')
    subplot_index = 1
    if zoomed_type is None:
        title = 'Event types\' rate (events per minute)' if title is None \
            else title
        fig.suptitle(title, size=16)
    else:
        pass
        # '{0} subtypes\' rate (events per minute)'.format(zoomed_type))

    for loc, rate_data_by_type in hfo_type_data_by_loc.items():
        elec_count = None
        axe = plt.subplot('{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        title = '{l}'.format(l=loc)
        plot_data = {type: {} for type in rate_data_by_type.keys()}
        for e_type, rate_data in rate_data_by_type.items():
            plot_data[e_type]['preds'] = rate_data['evt_rates']
            plot_data[e_type]['labels'] = rate_data['soz_labels']
            plot_data[e_type]['legend'] = short_name('{t}'.format(t=e_type))
            scores = {}
            if 'ec' in metrics:
                scores['ec'] = rate_data['evt_count']
            if 'pse' in metrics:
                scores['pse'] = rate_data['pse']
            if 'pnee' in metrics:
                scores['pnee'] = rate_data['pnee']
            if 'auc' in metrics:
                scores['AUC_ROC'] = round(rate_data['AUC_ROC'], 2)
            if 'Simulator' in e_type:
                scores['conf'] = rate_data['conf']
            plot_data[e_type]['scores'] = scores

            if elec_count is None:
                elec_count = rate_data['elec_count']
            elif elec_count != rate_data['elec_count']:
                print(
                    'Elec count of type {0}: {1}, other elec_count {2}'.format(
                        e_type, rate_data['elec_count'], elec_count))
                raise RuntimeError(
                    'Elec count disagreement among types in ROCs plot')

        if change_tab_path is None:
            tab_sav_path = None
        else:
            tab_sav_path = str(Path(Path(roc_saving_path).parent,
                                    loc.replace(' ', '_'),
                                    '{l}_table'.format(l=loc.replace(' ', '_'))
                                    ))
            Path(tab_sav_path).parent.mkdir(parents=True, exist_ok=True)

        # Plots axe for one location and all types

        last_row = np.ceil(subplot_index / cols) == np.ceil(
            subplot_count / cols)
        plot_x_label = True if subplot_count == 1 or last_row else False
        plot_y_label = True if subplot_count == 1 or subplot_index % cols == \
                               1 else False

        superimposed_rocs(plot_data, title, axe, elec_count, colors,
                          tab_sav_path, plot_x_label, plot_y_label)
        subplot_index += 1
    plt.subplots_adjust(wspace=0.3, hspace=0.35)
    for fmt in ['pdf', 'png', 'eps']:
        saving_path_f = '{file_path}.{format}'.format(file_path=roc_saving_path,
                                                      format=fmt)
        # if fmt == 'pdf':
        # print('ROC saving path: {0}'.format(saving_path_f))
        plt.savefig(saving_path_f, bbox_inches='tight')
    # plt.show()
    plt.close(fig)


def update_flex_point(flex_point_fpr, fpr):
    # fprs are ordered increasing order
    # fpr[-2] is the last real values since -1 is a hardcoded 1.
    flex_point_fpr = fpr[-2] if fpr[-2] > flex_point_fpr else flex_point_fpr
    return flex_point_fpr


def bootstrapped_scores(labels, preds, scoring_funcs, N=1000, ax=None, evt_type=None):
    np_labels = np.array(labels)
    np_preds = np.array(preds)
    n_bootstraps = N
    rng_seed = 42  # control reproducibility
    scores_by_name = {name: [] for name in scoring_funcs.keys()}
    rng = np.random.RandomState(rng_seed)
    fpr_its, tpr_its = [], []

    mean_fpr = np.linspace(0, 1, 100)
    flex_point_fpr = 0.05
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(np_preds), len(np_preds))
        if len(np.unique(np_labels[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        labels_it, pred_it = np_labels[indices], np_preds[indices]
        for score_name, score_func in scoring_funcs.items():
            score = score_func(labels_it, pred_it)
            scores_by_name[score_name].append(score)
            # print("Bootstrap #{} {}: {:0.3f}".format(i + 1, score_name, score))
        fpr_it, tpr_it, threshold = metrics.roc_curve(labels_it, pred_it)

        interp_tpr = np.interp(mean_fpr, fpr_it, tpr_it)
        interp_tpr[0] = 0.0
        tpr_its.append(interp_tpr)

        flex_point_fpr = update_flex_point(flex_point_fpr, fpr_it)

    mean_tpr = np.mean(tpr_its, axis=0)
    mean_tpr[-1] = 1.0

    std_tpr = np.std(tpr_its, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    scores_by_name['mean_fpr'], scores_by_name['mean_tpr'] = mean_fpr, mean_tpr
    scores_by_name['tpr_upper'], scores_by_name['tpr_lower'], scores_by_name[
        'std_tpr'] = tprs_upper, tprs_lower, std_tpr
    return scores_by_name, flex_point_fpr


# Reviewed
# Plots the ROCs of many types in a location given in plot_data, modifies the
# axe object
# It also may build tables of the global info in that location if you
# uncomment that piece of code

def replace_after_flex(flex_point_fpr, fpr, tpr, tpr_lower, tpr_upper, std_tpr):
    # replaces after flex point using last tpr
    tpr_plateau_idx = None
    for i, value in enumerate(fpr, start=0):
        if tpr_plateau_idx is not None:
            tpr[i] = tpr[tpr_plateau_idx]
            tpr_lower[i] = tpr[i] - std_tpr[tpr_plateau_idx]
            tpr_upper[i] = tpr[i] + std_tpr[tpr_plateau_idx]
            continue

        if fpr[i] < flex_point_fpr:
            continue
        else:
            tpr_plateau_idx = i


def superimposed_rocs(plot_data, title, axe, elec_count, colors=None,
                      tab_sav_path=None, plot_x_label=True, plot_y_label=True):
    if plot_x_label:
        axe.set_xlabel('FPR', fontdict={'fontsize': 12})
    if plot_y_label:
        axe.set_ylabel('TPR', fontdict={'fontsize': 12})  # uno menos q title
    location = title
    axe.set_title(title, fontdict={'fontsize': 14})
    legend_loc = (0.03, 0.67)  # 'best'
    # calculate the fpr and tpr for all thresholds of the classification
    roc_data, pses = [], []
    for evt_type, info in plot_data.items():
        labels, preds = info['labels'], info['preds']

        pauc_05_scorer = partial(metrics.roc_auc_score, max_fpr=0.05)
        pauc_10_scorer = partial(metrics.roc_auc_score, max_fpr=0.10)
        scoring_funcs = dict(
            pauc_05=pauc_05_scorer,
            pauc_10=pauc_10_scorer
        )

        scores_by_name, flex_point_fpr = bootstrapped_scores(labels, preds, scoring_funcs=scoring_funcs, N=1000, ax=axe,
                                                             evt_type=evt_type)

        for scores in scores_by_name.values():
            scores.sort()

        confidence = info['scores']['conf'] if 'Simulator' in evt_type else None
        if 'pse' in info['scores'].keys():
            pses.append(info['scores']['pse'])

        bootstrapped_curve = True
        if bootstrapped_curve:
            fpr, tpr = scores_by_name['mean_fpr'], scores_by_name['mean_tpr']
            tpr_upper, tpr_lower, std_tpr = scores_by_name['tpr_upper'], scores_by_name['tpr_lower'], scores_by_name[
                'std_tpr']

        else:
            fpr, tpr, threshold = metrics.roc_curve(labels, preds)

        replace_after_flex(flex_point_fpr, fpr, tpr, tpr_lower, tpr_upper, std_tpr)
        if axe is not None:
            axe.fill_between(
                fpr,
                tpr_lower,
                tpr_upper,
                color=color_for(evt_type),
                alpha=0.2,
            )

        roc_data.append((evt_type, fpr, tpr, info['scores']['AUC_ROC'], scores_by_name,
                         confidence))
    pauc_index = 4
    roc_data.sort(key=lambda x: np.mean(x[pauc_index]['pauc_05']), reverse=True)  # Orders descendent by pAUC mean

    if len(pses) > 0:
        pse = pses[0]
        for e in pses:
            if e != pse:
                # This shouldnt happen, soz electrodes should be independent of the type of them
                raise RuntimeError(
                    'SOZ electrode percentage disagreement among evt types of the same location')

    columns, rows = [], []
    i = 0

    # For each type
    for evt_type, fpr, tpr, auc, sorted_paucs, conf in roc_data:

        legend = plot_data[evt_type]['legend']

        add_pauc_05 = True
        if add_pauc_05:
            pauc_05_arr = sorted_paucs['pauc_05']
            pauc_05_mean = round(np.mean(pauc_05_arr), 3)
            confidence_lower_05 = round(pauc_05_arr[int(0.025 * len(pauc_05_arr))], 3)
            confidence_upper_05 = round(pauc_05_arr[int(0.975 * len(pauc_05_arr))], 3)

            legend = legend + ' pauc .05 {mean} CI ({low},{up})'.format(mean=pauc_05_mean,
                                                                        low=confidence_lower_05,
                                                                        up=confidence_upper_05)
        add_pauc_10 = False
        if add_pauc_10:
            pauc_10_arr = sorted_paucs['pauc_10']
            pauc_10_mean = round(np.mean(pauc_10_arr), 3)
            confidence_lower_10 = round(pauc_10_arr[int(0.025 * len(pauc_10_arr))], 3)
            confidence_upper_10 = round(pauc_10_arr[int(0.975 * len(pauc_10_arr))], 3)

            legend = legend + ' pauc .10 mean {mean} 95% CI ({low} - {up})'.format(mean=pauc_10_mean,
                                                                                   low=confidence_lower_10,
                                                                                   up=confidence_upper_10)
        index = 0
        for f in fpr:
            if f <= 0.2:
                index += 1
        if location == WHOLE_BRAIN_L0C:
            legend_loc = 'lower right'
            text_height = 0.9
            legend_size = 10
            axe.plot([0, 1], [0, 1], 'r--')
            axe.set_xlim([0, 1])
            axe.set_ylim([0, 1])
            axe.plot(fpr, tpr, color_for(evt_type) if colors is None else
            color_list[i], label=legend)
            # location for the zoomed portion
            sub_axes = plt.axes([.6, .6, .25, .25])
            sub_axes.plot([0, 0.2], [0, 0.2], 'r--')
            sub_axes.set_xlim([0, 0.2])
            sub_axes.set_ylim([0, 1])
            # plot the zoomed portion
            sub_axes.plot(fpr[:index + 1], tpr[:index + 1], color_for(evt_type) if colors is
                                                                                   None else
            color_list[i], label=legend)
            # insert the zoomed figure
            plt.setp(sub_axes)
        else:
            if location in ['Hippocampus', 'Amygdala']:
                text_height = 0.92  # 0.55
                legend_size = 8  # 8 14
            else:
                text_height = 0.92  # .55
                legend_size = 8  # 10 7 8

            partial_roc = True
            if partial_roc:
                axe.plot([0, 0.2], [0, 0.2], 'r--')
                axe.set_xlim([0, 0.2])
                axe.set_ylim([0, 1])
                axe.plot(fpr[:index + 1], tpr[:index + 1],
                         color_for(evt_type) if colors is None else color_list[i],
                         label=legend)
            else:
                axe.plot([0, 1], [0, 1], 'r--')
                axe.set_xlim([0, 1])
                axe.set_ylim([0, 1])
                axe.plot(fpr, tpr, color_for(evt_type) if colors is None else
                color_list[i], label=legend)

        # Building report tables
        # si agregas pAUC agregar columna
        rows.append([evt_type] + [str(plot_data[evt_type]['scores'][k]) for k in
                                  ['ec', 'pse', 'pnee', 'AUC_ROC'] if
                                  k in plot_data[evt_type]['scores'].keys()] + [pauc_05_mean])
        i += 1
    # axe.set_title(title, fontdict={'fontsize': text_size}, loc='left')
    info_text = 'EC: {0}'.format(elec_count)
    plot_pse_text = True
    if len(pses) > 0 and plot_pse_text:
        info_text = info_text + '\nPSE: {0}'.format(
            round(np.mean(pses), 2))
    axe.text(0.05, text_height, info_text, bbox=dict(facecolor='grey',
                                                     alpha=0.5),
             transform=axe.transAxes, fontsize=legend_size)

    axe.legend(loc=legend_loc, prop={'size': legend_size})

    columns = [title] + [k for k in ['ec', 'pse', 'pnee', 'AUC_ROC'] if
                         k in plot_data[evt_type][
                             'scores'].keys()] + ['pAUC']  # Title here is the
    # location
    if tab_sav_path is not None:
        pass
        # plot_score_in_loc_table(columns, rows, colors, tab_sav_path)


# 4 Machine learning ROCs
# Plots the ROC of the mean tpr and mean fpr
# Ver si se usa
def ml_training_plot(data_by_model, loc_name, hfo_type,
                     roc=True, pre_rec=False,
                     saving_dir=FIG_SAVE_PATH[4]['dir']):
    fig = plt.figure()
    fig.suptitle('SOZ HFO classfiers training in {0}'.format(loc_name),
                 fontsize=16)
    models_to_run = list(data_by_model.keys())
    plot_axe = axes_by_model(plt, models_to_run)
    for model_name in models_to_run:
        # Plot ROC curve
        curve_kind = 'ROC'
        labels = data_by_model[model_name]['y']
        probs = data_by_model[model_name]['probas']
        fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
        youden = youden(fpr, tpr, thresholds)
        plot_axe[model_name][curve_kind].plot(fpr, tpr, lw=1, alpha=0.8,
                                              label='AUC = %0.2f. Youden = %0.2f' %
                                                    (metrics.auc(fpr, tpr),
                                                     youden))
        plot_axe[model_name][curve_kind].plot([0, 1], [0, 1], linestyle='--',
                                              lw=2, color='r', label='Chance',
                                              alpha=.8)
        set_titles('False Positive Rate', 'True Postive Rate', model_name,
                   plot_axe[model_name][curve_kind])

        # PRE REC
        precision, recall, thresholds = metrics.precision_recall_curve(labels,
                                                                       probs)
        precision = np.array(list(reversed(list(precision))))
        recall = np.array(list(reversed(list(recall))))
        # thesholds = np.array(list(reversed(list(thresholds))))
        ap = metrics.average_precision_score(labels, probs)
        auc_val = metrics.auc(recall, precision)
        curve_kind = 'PRE_REC'
        plot_axe[model_name][curve_kind].plot(recall, precision, color='b',
                                              label='AUC = %0.2f. AP = %0.2f' % (
                                                  auc_val, ap),
                                              lw=2, alpha=.8)
        set_titles('Recall', 'Precision', model_name,
                   plot_axe[model_name][curve_kind])

    # Saving the figure
    saving_path = str(Path(saving_dir, loc_name, hfo_type, 'ml_train_plot'))
    for fmt in ['pdf', 'png']:
        saving_path_f = '{file_path}.{format}'.format(file_path=saving_dir,
                                                      format=fmt)
        if fmt == 'pdf':
            print('ROC saving path: {0}'.format(saving_path_f))
        plt.savefig(saving_path_f, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def feature_importances(feature_list, importances, hfo_type_name):
    fig = plt.figure()
    axe = plt.subplot(111)
    # plt.style.use('fivethirtyeight')

    # Vertical bars
    # x_values = list(range(len(importances)))
    # axe.bar(importances, x_values, orientation='horizontal')
    # plt.xticks(x_values, feature_list ) #rotation='vertical'

    # Horizontal bars
    pos = np.arange(len(feature_list))
    rects = axe.barh(pos, importances,
                     align='center',
                     height=0.5,
                     tick_label=feature_list)
    axe.set_title('{0} Variable Importances'.format(hfo_type_name))
    axe.set_ylabel('Importance')
    axe.set_xlabel('Variable')
    plt.show()


# xgb.plot_importance(xg_reg)
# plt.rcParams['figure.figsize'] = [5, 5]

# Auxiliary functions reviewed

def color_for(t):
    if 'Validation baseline' == t:
        return 'blue'
    if 'Youden' in t:
        return 'magenta'

    if 'FPR' in t:
        colors = ['darkred', 'firebrick', 'red', 'indianred', 'lightcoral',
                  'aquamarine', 'springgreen', 'limegreen', 'green',
                  'darkgreen']
        return colors[int(t[-1])]

    if t == 'HFOs':
        return 'b'
    if t == 'RonO+RonS+Fast RonO+Fast RonS':
        return 'b'
    if t == 'Fast RonO+Fast RonS+RonO+RonS':
        return 'b'
    if t == 'RonO':
        return 'b'
    if t == 'RonS':
        return 'g'
    if t == 'Fast RonO':
        return 'm'
    if t == 'Fast RonS':
        return 'y'
    if t == 'Spikes':
        return 'c'
    if t == 'Sharp Spikes':
        return 'k'
    if t == 'Spikes + Sharp Spikes':
        return 'magenta'
    if t == 'Filtered RonO':
        return 'mediumslateblue'
    if t == 'Filtered RonS':
        return 'lime'
    if t == 'Filtered Fast RonO':
        return 'darkviolet'
    if t == 'Filtered Fast RonS':
        return 'gold'

    raise ValueError('graphics.color_for is undefined for type: {0}'.format(t))


def table_color_for(t):
    if t == 'HFOs':
        return 'blue'
    if t == 'RonO+RonS+Fast RonO+Fast RonS':
        return 'blue'
    if t == 'RonO':
        return 'blue'
    if t == 'RonS':
        return 'green'
    if t == 'Fast RonO':
        return 'magenta'
    if t == 'Fast RonS':
        return 'yellow'
    if t == 'Spikes':
        return 'lightcyan'
    if t == 'Sharp Spikes':
        return 'black'
    if t == 'Spikes + Sharp Spikes':
        return 'magenta'
    if t == 'Filtered RonO':
        return 'mediumslateblue'
    if t == 'Filtered RonS':
        return 'lime'
    if t == 'Filtered Fast RonO':
        return 'darkviolet'
    if t == 'Filtered Fast RonS':
        return 'gold'
    else:
        return 'white'
        # raise ValueError('graphics.table_color_for is undefined for type: {
        # 0}'.format(t))


def color_for_scatter(granularity, type, hip=False):
    color_by_type = {
        'RonS': {2: 'darkgreen', 3: 'mediumseagreen', 5: 'lime',
                 'Hippocampus': 'black'},
        'RonO': {2: 'midnightblue', 3: 'mediumblue', 5: 'cornflowerblue',
                 'Hippocampus': 'dimgrey'},
        'Fast RonO': {2: 'darkorange', 3: 'orange', 5: 'wheat',
                      'Hippocampus': 'darkgray'},
        'Fast RonS': {2: 'darkred', 3: 'red', 5: 'lightcoral',
                      'Hippocampus': 'lightgray'}
    }
    try:
        if hip:
            return color_by_type[type]['Hippocampus']
        else:
            return color_by_type[type][3]
    except Exception as e:
        print('TYPE {0} GRANULARITY {1}'.format(type, granularity))
        raise RuntimeError('Undefined color for granularity and type')


def color_by_gran(granularity):
    if granularity == 0:
        return 'lightblue'
    elif granularity == 2:
        return 'lightsalmon'
    elif granularity == 3:
        return 'lightgreen'
    elif granularity == 5:
        return 'lightyellow'
    else:
        raise RuntimeError(
            'Undefined color for granularity {0}'.format(granularity))


def orca_save(fig, saving_path, width=None, height=None, scale=None):
    if Path(ORCA_EXECUTABLE).exists():
        #  print('Orca executable_path: {0}'.format(plotly.io.orca.config.executable))
        plotly.io.orca.config.executable = ORCA_EXECUTABLE
        plotly.io.orca.config.save()
        #  print('Orca executable_path after save: {0}'.format(
        #  plotly.io.orca.config.executable))
        try:
            fig.write_image(saving_path + '.pdf', width=width, height=height,
                            scale=scale)
            fig.write_html(saving_path + '.html')
        except ValueError as err:
            print('Orca executable is probably invalid, save figure manually.')
            raise err
    else:
        print('You need to install orca and define orca executable to save '
              'this figure.')
        # You can install orca in ubuntu with npm install -g electron@6.1.4 orca


# Reviewed
def axes_by_model(plt, models_to_run):
    subplot_count = len(models_to_run) * 2
    if subplot_count == 2:
        rows = 2
        cols = 1
    elif subplot_count == 4:
        rows = 2
        cols = 2
    elif subplot_count == 6:
        rows = 2
        cols = 3
    elif subplot_count == 8:
        rows = 2
        cols = 4
    else:
        raise RuntimeError('Subplot count not implemented')
    axes = {}
    subplot_index = 1
    for m in models_to_run:
        axes[m] = dict()
        axes[m]['ROC'] = plt.subplot(
            '{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        subplot_index += 1
    for m in models_to_run:
        axes[m]['PRE_REC'] = plt.subplot(
            '{r}{c}{i}'.format(r=rows, c=cols, i=subplot_index))
        subplot_index += 1
    return axes


# Reviewed
def set_titles(x_title, y_title, model_name, axe):
    axe.set_xlim([-0.05, 1.05])
    axe.set_ylim([-0.05, 1.05])
    axe.set_xlabel(x_title)
    axe.set_ylabel(y_title)
    axe.set_title(model_name)
    axe.legend(loc="lower right")


def k_means_clusters_plot(data):
    x = data['freq_av'] if 'freq_av' in data.keys() else data['freq_pk']
    y = data['duration']
    z = data['power_av'] if 'power_av' in data.keys() else data['power_pk']
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter3D(x, y, z, c=x, cmap='hsv', marker='o')
    ax.set_xlabel('Freq av' if 'freq_av' in data.keys() else 'Freq pk')
    ax.set_ylabel('Duration (ms)')
    ax.set_zlabel('Power av' if 'power_av' in data.keys() else 'Power pk')

    ax.set_title('FronO clusters')
    ax.legend(loc="lower right")

    plt.show()


# AUX

def pretty_print(feature):
    if feature == 'HFO_rate':
        return 'HFO rate per minute (log10)'
    elif feature == 'duration':
        return 'Duration (ms)'
    elif feature == 'power_pk':
        return 'Peak Power (log10)'
    elif feature == 'freq_pk':
        return 'Peak Frequency'
    else:
        return feature.capitalize()


def rem_lobe(location):
    if 'Lobe' in location:
        return location[:-len('Lobe')]
    else:
        return location


'''
# For generating independent histogram per type
def plot_feature_distribution(soz_data, nsoz_data, feature, type, stats,
                              test_names, saving_dir):
    saving_dir = str(Path(saving_dir, feature, type))
    Path(saving_dir).mkdir(parents=True, exist_ok=True)
    fig_path = str(Path(saving_dir, feature + '_distr.pdf'))
    # print('Distribution saving path: {0}'.format(fig_path))
    # print('Plotting feature:{f} for type:{t}'.format(
    #    f=feature, t=type))
    sns.set_style("white")
    # Plot
    kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})

    fig = plt.figure(figsize=(10, 7), dpi=80)
    fig.suptitle('{feat} SOZ vs NSOZ distributions'.format(
        feat=feature.capitalize()),
        fontsize=20)
    plt.xlabel(feature.capitalize(), fontsize=18)
    plt.ylabel('Frequency', fontsize=16)
    sns.distplot(soz_data, color="red", label="SOZ", **kwargs)
    sns.distplot(nsoz_data, color="green", label="NSOZ", **kwargs)
    axes = fig.gca()
    # Prints stats in figure
    X = {'D':0.05, 'W':0.35, 'U':0.65}
    for S_name in test_names.keys():
        S_val = round(stats[feature][type][test_names[S_name]][S_name], 4)
        S_pval = format(stats[feature][type][test_names[S_name]]['pval'],'.2e')
        # For text block coords (0, 0) is bottom and (1, 1) is top
        axes.text(x=X[S_name],
                  y=0.88,
                  s='{t_name}\n{S_name}: {S_val} \npVal: {S_pval}'.format(
                      t_name=test_names[S_name],
                      S_name=S_name,
                      S_val=S_val,
                      S_pval=S_pval),
                  bbox=dict(facecolor='grey', alpha=0.5),
                  transform=axes.transAxes, fontsize=12)
        #plt.xlim(50, 75)
    plt.legend(loc='lower right')  # 'lower right'
    fig.savefig(fig_path)
    plt.close(fig)
    # plt.show()
'''
