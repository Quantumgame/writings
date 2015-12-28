import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasp.plots import multi_plot

from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder


def clean_region(reg):
    if '-' in reg:
        return '?'
    if reg.startswith('L2'):
        return 'L2'
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

    nbands = len(freqs)
    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])

    for (bird,block,segment,hemi),gdf in g:

        wtup = (bird,block,segment,hemi)
        index2electrode = agg.index2electrode[wtup]
        cell_index2electrode = agg.cell_index2electrode[wtup]

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

            # get the region for this electrode
            i = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert i.sum() == 1
            region = clean_region(edata.region[i].values[0])

            # append the single electrode performances and likelihood ratios to the single electrode dataset
            single_electrode_data['bird'].append(bird)
            single_electrode_data['block'].append(block)
            single_electrode_data['segment'].append(segment)
            single_electrode_data['hemi'].append(hemi)
            single_electrode_data['electrode'].append(e)
            single_electrode_data['region'].append(region)

            for aprop in anames:
                # append single electrode peformance
                single_electrode_data['perf_%s' % aprop].append(perfs['perf_%s' % aprop])
                # append likelihood ratio
                full_likelihood = band0_perfs['lk_%s_%s' % (aprop, 'lfp')]
                leave_one_out_likelihood = perfs_exel['lk_%s' % aprop]
                lkrat = 2*(leave_one_out_likelihood - full_likelihood)
                single_electrode_data['lkrat_%s' % aprop].append(lkrat)

    df_me = pd.DataFrame(multi_electrode_data)
    df_me.to_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'), index=False)

    df_se = pd.DataFrame(single_electrode_data)
    df_se.to_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_perfs.csv'), index=False)

    return df_me


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

        plt.plot(pdata['freqs'], pdata['lkrat_lfp'], 'k-', linewidth=3.0, alpha=0.7)
        plt.plot(pdata['freqs'], pdata['lkrat_spike'], 'k--', linewidth=3.0, alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Likelihood Ratio')
        plt.legend(['LFP', 'Spike'], fontsize='x-small')
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

    # df_me = export_dfs(agg)
    # df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))
    df_se = pd.read_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_perfs.csv'))

    plist = list()
    g = df_se.groupby(['bird', 'block', 'segment', 'hemi'])
    for gkey,gdf in g:
        plist.append({'perf':gdf.perf_q2, 'lkrat':gdf.lkrat_q2, 't':'_'.join(gkey)})

    def _plot_scat(pdata, ax):
        plt.sca(ax)
        cc = np.corrcoef(pdata['perf'], pdata['lkrat'])[0, 1]
        plt.plot(pdata['perf'], pdata['lkrat'], 'co')
        plt.title('cc=%0.2f, %s' % (cc, pdata['t']), fontsize=8)
        plt.axis('tight')

    multi_plot(plist, _plot_scat, nrows=5, ncols=7)

    # draw_perf_hists(agg, df_me)
    # draw_freq_lkrats(agg, df_me)
    plt.show()

if __name__ == '__main__':
    draw_figures()

