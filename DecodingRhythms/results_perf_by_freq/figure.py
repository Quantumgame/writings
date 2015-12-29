import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

from DecodingRhythms.utils import set_font
from lasp.plots import multi_plot, custom_legend, grouped_boxplot
from utils import get_this_dir
from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder


def clean_region(reg):
    if '-' in reg:
        return '?'
    if reg.startswith('L2'):
        return 'L2'
    if reg == 'CM':
        return '?'
    return reg


def export_dfs(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs = agg.freqs
    # read electrode data
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    # initialize multi electrode dataset dictionary
    multi_electrode_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'band':list()}
    anames = agg.acoustic_props + ['category']
    for aprop in anames:
        for t in ['lfp', 'spike']:
            multi_electrode_data['perf_%s_%s' % (aprop, t)] = list()
            multi_electrode_data['lkrat_%s_%s' % (aprop, t)] = list()

    # initialize single electrode dataset dictionary
    single_electrode_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(),
                             'region':list()}

    anames = agg.acoustic_props + ['category']
    for aprop in anames:
        single_electrode_data['perf_%s' % aprop] = list()
        single_electrode_data['lkrat_%s' % aprop] = list()

    # initialize single cell dataset dictionary
    cell_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(),
                 'region':list(), 'cell_index':list()}
    for aprop in anames:
        cell_data['perf_%s' % aprop] = list()
        cell_data['lkrat_%s' % aprop] = list()

    nbands = len(freqs)
    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])

    for (bird,block,segment,hemi),gdf in g:

        wtup = (bird,block,segment,hemi)
        index2electrode = agg.index2electrode[wtup]
        cell_index2electrode = agg.cell_index2electrode[wtup]

        # get the region by electrode
        electrode2region = dict()
        for e in index2electrode:
            i = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert i.sum() == 1
            electrode2region[e] = clean_region(edata.region[i].values[0])

        # collect multi-electrode multi-band dataset
        band0_perfs = None
        for b in range(nbands+1):

            exfreq = b > 0

            perfs = dict()
            anames = agg.acoustic_props + ['category']
            for aprop in anames:
                for t,decomp in [('lfp', 'locked'), ('spike', 'spike_psd')]:
                    # get multielectrode LFP decoder performance
                    i = (gdf.e1 == -1) & (gdf.e2 == -1) & (gdf.cell_index == -1) & (gdf.band == b) & (gdf.exfreq == exfreq) & \
                        (gdf.exel == False) & (gdf.aprop == aprop) & (gdf.decomp == decomp)
                    assert i.sum() == 1, "Zero or more than 1 result for (%s, %s, %s, %s), decomp=locked, band=%d: i.sum()=%d" % (bird, block, segment, hemi, b, i.sum())
                    if aprop == 'category':
                        perfs['perf_%s_%s' % (aprop, t)] = gdf.pcc[i].values[0]
                    else:
                        perfs['perf_%s_%s' % (aprop, t)] = gdf.r2[i].values[0]

                    lk = gdf.likelihood[i].values[0]
                    if aprop == 'category':
                        nsamps = gdf.num_samps[i].values[0]
                        lk *= nsamps
                    perfs['lk_%s_%s' % (aprop, t)] = lk

            multi_electrode_data['bird'].append(bird)
            multi_electrode_data['block'].append(block)
            multi_electrode_data['segment'].append(segment)
            multi_electrode_data['hemi'].append(hemi)
            multi_electrode_data['band'].append(b)
            for k,v in perfs.items():
                if k.startswith('perf'):
                    multi_electrode_data[k].append(v)

            if b == 0:
                band0_perfs = perfs
                for aprop in anames:
                    for t in ['lfp', 'spike']:
                        multi_electrode_data['lkrat_%s_%s' % (aprop,t)].append(0)
            else:
                # compute the likelihood ratio for each acoustic property on this band
                for aprop in anames:
                    for t in ['lfp', 'spike']:
                        full_likelihood = band0_perfs['lk_%s_%s' % (aprop, t)]
                        leave_one_out_likelihood = perfs['lk_%s_%s' % (aprop, t)]
                        lkrat = 2*(leave_one_out_likelihood - full_likelihood)
                        multi_electrode_data['lkrat_%s_%s' % (aprop, t)].append(lkrat)

        # collect single electrode dataset
        for e in index2electrode:

            # get LFP performance data for this electrode, with and without leave-one-out (the variable "exel")
            perfs = dict()
            perfs_exel = dict()
            anames = agg.acoustic_props + ['category']
            for aprop in anames:
                for exel in [True, False]:
                    p = perfs
                    if exel:
                        p = perfs_exel
                    # get multielectrode LFP decoder performance
                    i = (gdf.e1 == e) & (gdf.e2 == e) & (gdf.cell_index == -1) & (gdf.band == 0) & (gdf.exfreq == False) & \
                        (gdf.exel == exel) & (gdf.aprop == aprop) & (gdf.decomp == 'locked')
                    assert i.sum() == 1, "Zero or more than 1 result for (%s, %s, %s, %s), decomp=locked, e=%d: i.sum()=%d" % (bird, block, segment, hemi, e, i.sum())
                    if aprop == 'category':
                        p['perf_%s' % aprop] = gdf.pcc[i].values[0]
                    else:
                        p['perf_%s' % aprop] = gdf.r2[i].values[0]

                    lk = gdf.likelihood[i].values[0]
                    if aprop == 'category':
                        nsamps = gdf.num_samps[i].values[0]
                        lk *= nsamps
                    p['lk_%s' % aprop] = lk

            # append the single electrode performances and likelihood ratios to the single electrode dataset
            single_electrode_data['bird'].append(bird)
            single_electrode_data['block'].append(block)
            single_electrode_data['segment'].append(segment)
            single_electrode_data['hemi'].append(hemi)
            single_electrode_data['electrode'].append(e)
            single_electrode_data['region'].append(electrode2region[e])

            for aprop in anames:
                # append single electrode peformance
                single_electrode_data['perf_%s' % aprop].append(perfs['perf_%s' % aprop])
                # append likelihood ratio
                full_likelihood = band0_perfs['lk_%s_%s' % (aprop, 'lfp')]
                leave_one_out_likelihood = perfs_exel['lk_%s' % aprop]
                lkrat = 2*(leave_one_out_likelihood - full_likelihood)
                single_electrode_data['lkrat_%s' % aprop].append(lkrat)

        # collect single cell dataset
        for e in index2electrode:

            # count the number of cells and get their indices
            i = (gdf.e1 == e) & (gdf.e2 == e) & (gdf.cell_index != -1) & (gdf.band == 0) & (gdf.exfreq == False) & \
                    (gdf.exel == False) & (gdf.decomp == 'spike_psd')
            if i.sum() == 0:
                print 'No cells for (%s, %s, %s, %s), e=%d' % (bird, block, segment, hemi, e)
                continue

            cell_indices = sorted(gdf[i].cell_index.unique())
            for ci in cell_indices:

                missing_data = False
                # get cell performance data for this electrode, with and without leave-one-out (the variable "exel")
                perfs = dict()
                perfs_exel = dict()
                anames = agg.acoustic_props + ['category']
                for aprop in anames:
                    for exel in [True, False]:
                        p = perfs
                        if exel:
                            p = perfs_exel

                        # get multielectrode LFP decoder performance
                        i = (gdf.e1 == e) & (gdf.e2 == e) & (gdf.cell_index == ci) & (gdf.band == 0) & (gdf.exfreq == False) & \
                            (gdf.exel == exel) & (gdf.aprop == aprop) & (gdf.decomp == 'spike_psd')
                        if i.sum() == 0:
                            print "No result for (%s, %s, %s, %s), decomp=spike_psd, e=%d, ci=%d: i.sum()=%d" % (bird, block, segment, hemi, e, ci, i.sum())
                            missing_data = True
                            continue
                        if i.sum() > 1:
                            print "More than 1 result for (%s, %s, %s, %s), decomp=spike_psd, e=%d, ci=%d: i.sum()=%d" % (bird, block, segment, hemi, e, ci, i.sum())
                            missing_data = True
                            continue

                        if aprop == 'category':
                            p['perf_%s' % aprop] = gdf.pcc[i].values[0]
                        else:
                            p['perf_%s' % aprop] = gdf.r2[i].values[0]

                        lk = gdf.likelihood[i].values[0]
                        if aprop == 'category':
                            nsamps = gdf.num_samps[i].values[0]
                            lk *= nsamps
                        p['lk_%s' % aprop] = lk

                if missing_data:
                    print 'Skipping cell %d on electrode %d for (%s, %s, %s, %s)' % (ci, e, bird, block, segment, hemi)
                    continue

                # append the single electrode performances and likelihood ratios to the single electrode dataset
                cell_data['bird'].append(bird)
                cell_data['block'].append(block)
                cell_data['segment'].append(segment)
                cell_data['hemi'].append(hemi)
                cell_data['electrode'].append(e)
                cell_data['region'].append(electrode2region[e])
                cell_data['cell_index'].append(ci)

                for aprop in anames:
                    # append single electrode peformance
                    cell_data['perf_%s' % aprop].append(perfs['perf_%s' % aprop])
                    # append likelihood ratio
                    full_likelihood = band0_perfs['lk_%s_%s' % (aprop, 'spike')]
                    leave_one_out_likelihood = perfs_exel['lk_%s' % aprop]
                    lkrat = 2*(leave_one_out_likelihood - full_likelihood)
                    cell_data['lkrat_%s' % aprop].append(lkrat)

    df_me = pd.DataFrame(multi_electrode_data)
    df_me.to_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'), index=False)

    df_se = pd.DataFrame(single_electrode_data)
    df_se.to_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_perfs.csv'), index=False)

    df_cell = pd.DataFrame(cell_data)
    df_cell.to_csv(os.path.join(data_dir, 'aggregate', 'cell_perfs.csv'), index=False)

    return df_me,df_se,df_cell


def draw_acoustic_perf_boxplots(agg, df_me):

    aprops_to_display = ['maxAmp', 'sal', 'meanspect', 'q1', 'q2', 'q3',
                         'entropyspect', 'meantime', 'entropytime']

    df0 = df_me[df_me.band == 0]

    bp_data = dict()
    for aprop in aprops_to_display:
        lfp_perfs = df0['perf_%s_lfp' % aprop].values
        spike_perfs = df0['perf_%s_spike' % aprop].values
        bp_data[aprop] = [lfp_perfs, spike_perfs]

    figsize = (24, 10)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    grouped_boxplot(bp_data, group_names=aprops_to_display, subgroup_names=['LFP', 'Spike'],
                    subgroup_colors=['#0068A5', '#F0DB00'], box_spacing=1.5)
    plt.xlabel('Acoustic Feature')
    plt.ylabel('Decoder R2')

    fname = os.path.join(get_this_dir(), 'perf_boxplots.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_perf_hists(agg, df_me):

    assert isinstance(agg, AggregateLFPAndSpikePSDDecoder)
    freqs = agg.freqs

    aprops_to_display = ['category', 'maxAmp', 'meanspect', 'stdspect', 'q1', 'q2', 'q3', 'skewspect', 'kurtosisspect',
                         'sal', 'entropyspect', 'meantime', 'stdtime', 'entropytime']

    # make histograms of performances across sites for each acoustic property
    perf_list = list()
    i = (df_me.band == 0)
    for aprop in aprops_to_display:
        lfp_perf = df_me[i]['perf_%s_%s' % (aprop, 'lfp')].values
        spike_perf = df_me[i]['perf_%s_%s' % (aprop, 'spike')].values

        perf_list.append({'lfp_perf':lfp_perf, 'spike_perf':spike_perf,
                          'lfp_mean':lfp_perf.mean(), 'spike_mean':spike_perf.mean(),
                          'aprop':aprop})

    # make plots
    print 'len(perf_list)=%d' % len(perf_list)

    def _plot_hist(pdata, ax):
        plt.sca(ax)
        plt.hist(pdata['lfp_perf'], bins=10, color='#00639E', alpha=0.7)
        plt.hist(pdata['spike_perf'], bins=10, color='#F1DE00', alpha=0.7)
        plt.legend(['LFP', 'Spike'], fontsize='x-small')
        plt.title(pdata['aprop'])
        if pdata['aprop'] == 'category':
            plt.xlabel('PCC')
        else:
            plt.xlabel('R2')
        plt.axis('tight')

    multi_plot(perf_list, _plot_hist, nrows=3, ncols=5, hspace=0.30, wspace=0.30)

    def _plot_scatter(pdata, ax):
        # pmax = max(pdata['lfp_perf'].max(), pdata['spike_perf'].max())
        pmax = 0.8
        plt.sca(ax)
        plt.plot(np.linspace(0, pmax, 20), np.linspace(0, pmax, 20), 'k-')
        plt.plot(pdata['lfp_perf'], pdata['spike_perf'], 'ko', alpha=0.7, markersize=10.)
        plt.title(pdata['aprop'])
        pstr = 'R2'
        if pdata['aprop'] == 'category':
            pstr = 'PCC'
        plt.xlabel('LFP %s' % pstr)
        plt.ylabel('Spike %s' % pstr)
        plt.axis('tight')
        plt.xlim(0, pmax)
        plt.ylim(0, pmax)

    multi_plot(perf_list, _plot_scatter, nrows=3, ncols=5, hspace=0.30, wspace=0.30)


def draw_freq_lkrats(agg, df_me):

    aprops_to_display = ['category', 'maxAmp', 'meanspect', 'stdspect', 'q1', 'q2', 'q3', 'skewspect', 'kurtosisspect',
                         'sal', 'entropyspect', 'meantime', 'stdtime', 'entropytime']

    assert isinstance(agg, AggregateLFPAndSpikePSDDecoder)
    freqs = agg.freqs
    nbands = len(freqs)

    x = np.linspace(1, 150, 1000)
    dof = 16
    p = chi2.pdf(x, dof)
    sig_thresh = min(x[p > 0.01])
    print 'sig_thresh=%f' % sig_thresh

    # make histograms of performances across sites for each acoustic property
    perf_list = list()
    for aprop in aprops_to_display:

        lkrat_by_band_lfp = np.zeros([len(freqs)])
        lkrat_by_band_spike = np.zeros([len(freqs)])
        std_by_band_lfp = np.zeros([len(freqs)])
        std_by_band_spike = np.zeros([len(freqs)])
        for b in range(1, nbands+1):
            i = df_me.band == b

            lkrats_lfp = df_me[i]['lkrat_%s_%s' % (aprop, 'lfp')].values
            lkrats_spike = df_me[i]['lkrat_%s_%s' % (aprop, 'spike')].values

            lkrat_by_band_lfp[b-1] = lkrats_lfp.mean()
            lkrat_by_band_spike[b-1] = lkrats_spike.mean()
            std_by_band_lfp[b-1] = lkrats_lfp.std(ddof=1)
            std_by_band_spike[b-1] = lkrats_spike.std(ddof=1)

        perf_list.append({'lkrat_lfp':lkrat_by_band_lfp, 'lkrat_spike':lkrat_by_band_spike,
                          'std_lfp':lkrat_by_band_lfp, 'std_spike':lkrat_by_band_spike,
                          'aprop':aprop, 'freqs':freqs})

    def _plot_freqs(pdata, ax):
        plt.sca(ax)
        if pdata['aprop'] == 'category':
            pass
        else:
            plt.axhline(0, c='k')

        gray = '#3c3c3c'
        plt.plot(pdata['freqs'], pdata['lkrat_lfp'], 'k-', linewidth=3.0, alpha=0.9)
        plt.plot(pdata['freqs'], pdata['lkrat_spike'], '--', c=gray, linewidth=3.0, alpha=0.9)
        plt.axhline(sig_thresh, c='k', alpha=1.0, linestyle='-', linewidth=2.0)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Likelihood Ratio')
        leg = custom_legend(['k', gray], ['LFP', 'Spike'], linestyles=['solid', 'dashed'])
        plt.legend(handles=leg, fontsize='x-small')
        plt.title(pdata['aprop'])
        plt.axis('tight')

        if pdata['aprop'] == 'category':
            pass
        else:
            plt.axhline(0, c='k')
            plt.ylim(0, 50)

    multi_plot(perf_list, _plot_freqs, nrows=3, ncols=5, hspace=0.30, wspace=0.30)


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])
    print '# of groups: %d' % len(g)

    # df_me,df_se,df_cell = export_dfs(agg)
    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))
    df_se = pd.read_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_perfs.csv'))

    # draw_perf_hists(agg, df_me)
    # draw_freq_lkrats(agg, df_me)
    draw_acoustic_perf_boxplots(agg, df_me)
    plt.show()

if __name__ == '__main__':
    set_font()
    draw_figures()

