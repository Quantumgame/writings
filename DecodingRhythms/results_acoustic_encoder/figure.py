import os
from copy import deepcopy, copy

import h5py
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from lasp.plots import custom_legend, grouped_boxplot, multi_plot, boxplot_with_colors

from DecodingRhythms.utils import set_font, get_this_dir, clean_region, COLOR_RED_SPIKE_RATE, COLOR_BLUE_LFP, \
    COLOR_PURPLE_LFP_CROSS, COLOR_CRIMSON_SPIKE_SYNC
from lasp.colormaps import magma
from zeebeez.aggregators.biosound import AggregateBiosounds

from zeebeez.aggregators.acoustic_encoder_decoder import AcousticEncoderDecoderAggregator
from zeebeez.utils import REDUCED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, \
    ACOUSTIC_FEATURE_COLORS, USED_ACOUSTIC_PROPS, ACOUSTIC_PROP_NAMES


def export_pairwise_encoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs,lags = get_freqs_and_lags()

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
            'electrode1':list(), 'electrode2':list(), 'regions':list(),
            'site':list(), 'lag':list(), 'r2':list(), 'dist':list()}

    weight_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
                   'electrode1':list(), 'electrode2':list(), 'regions':list(),
                   'site':list(), 'lag':list(), 'aprop':list(), 'w':list(), 'dist':list()}

    decomp = 'self+cross_locked'
    i = agg.df.decomp == decomp

    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird,block,seg,hemi),gdf in g:

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        index2electrode = agg.index2electrode[wkey]

        eperf = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]
        # normalize weights!
        eweights /= np.abs(eweights).max()

        site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

        for k,e1 in enumerate(index2electrode):

            regi = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e1)
            assert regi.sum() == 1
            reg1 = clean_region(edata[regi].region.values[0])

            eloc1 = np.array([edata[regi].dist_midline.values[0], edata[regi].dist_l2a.values[0]])

            for j in range(k):
                e2 = index2electrode[j]

                regi = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e2)
                assert regi.sum() == 1
                reg2 = clean_region(edata[regi].region.values[0])

                eloc2 = np.array([edata[regi].dist_midline.values[0], edata[regi].dist_l2a.values[0]])

                # compute the distance between electrodes in anatomical coordinates
                edist = np.linalg.norm(eloc1 - eloc2)

                for li,lag in enumerate(lags):

                    r2 = eperf[k, j, li]

                    if lag < 0:
                        regs = '%s->%s' % (reg2, reg1)
                    else:
                        regs = '%s->%s' % (reg1, reg2)

                    data['bird'].append(bird)
                    data['block'].append(block)
                    data['segment'].append(seg)
                    data['hemi'].append(hemi)
                    data['electrode1'].append(e1)
                    data['electrode2'].append(e2)
                    data['regions'].append(regs)
                    data['site'].append(site)
                    data['lag'].append(int(lag))
                    data['r2'].append(r2)
                    data['dist'].append(edist)

                    for ai,aprop in enumerate(USED_ACOUSTIC_PROPS):
                        w = eweights[k, j, li, ai]

                        weight_data['bird'].append(bird)
                        weight_data['block'].append(block)
                        weight_data['segment'].append(seg)
                        weight_data['hemi'].append(hemi)
                        weight_data['electrode1'].append(e1)
                        weight_data['electrode2'].append(e2)
                        weight_data['regions'].append(regs)
                        weight_data['site'].append(site)
                        weight_data['lag'].append(int(lag))
                        weight_data['aprop'].append(aprop)
                        weight_data['w'].append(w)
                        weight_data['dist'].append(edist)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, 'aggregate', 'pairwise_encoder_perfs_for_glm.csv'), header=True, index=False)

    wdf = pd.DataFrame(weight_data)
    wdf.to_csv(os.path.join(data_dir, 'aggregate', 'pairwise_encoder_weights_for_glm.csv'), header=True, index=False)


def plot_avg_pairwise_encoder_weights(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs,lags = get_freqs_and_lags()
    bs_agg = AggregateBiosounds.load(os.path.join(data_dir, 'aggregate', 'biosound.h5'))

    print agg.df.decomp.unique()
    decomp = 'full_psds+full_cfs'
    i = agg.df.decomp == decomp
    assert i.sum() > 0

    W = list()
    wdata = {'lag':list(), 'xindex':list()}
    nlags = len(lags)

    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird,block,seg,hemi),gdf in g:
        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        index2electrode = agg.index2electrode[wkey]

        eperf = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]

        # site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

        for k,e1 in enumerate(index2electrode):
            for j in range(k):
                for l,lag in enumerate(lags):
                    w = eweights[k, j, l, :]
                    wdata['lag'].append(int(lag))
                    wdata['xindex'].append(len(W))
                    W.append(w)

    wdf = pd.DataFrame(wdata)
    W = np.array(W)
    W[np.isnan(W)] = 0.
    W2 = W**2

    print 'nlags=%d' % nlags

    W2_by_lag = np.zeros([len(USED_ACOUSTIC_PROPS), nlags])
    for l,lag in enumerate(lags):
        i = wdf.lag == int(lag)
        assert i.sum() > 0
        ii = wdf.xindex[i].values
        W2_by_lag[:, l] = W2[ii, :].mean(axis=0)

    plot = True
    if plot:
        plt.figure()
        # ax = plt.subplot(1, 3, 1)
        absmax = W2_by_lag.max()
        plt.imshow(W2_by_lag, interpolation='nearest', aspect='auto', cmap=magma, vmin=0, vmax=absmax)
        plt.colorbar()
        plt.title('Mean Weights')
        plt.yticks(np.arange(len(USED_ACOUSTIC_PROPS)), USED_ACOUSTIC_PROPS)
        plt.xticks(np.arange(nlags), ['%d' % x for x in lags])

        plt.show()

    return lags,W2_by_lag,None


def export_psd_encoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs,lags = get_freqs_and_lags()

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(), 'cell_index':list(),
            'region':list(), 'site':list(), 'freq':list(), 'r2':list()}

    decomps = ['full_psds', 'spike_rate']

    assert isinstance(agg, AcousticEncoderDecoderAggregator)

    for decomp in decomps:

        i = agg.df.decomp == decomp
        g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
        for (bird,block,seg,hemi),gdf in g:

            assert len(gdf) == 1

            wkey = gdf['wkey'].values[0]
            eperf = agg.encoder_perfs[wkey]
            index2electrode = agg.index2electrode[wkey]
            cell_index2electrode = agg.cell_index2electrode[wkey]

            site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

            if decomp.endswith('psds'):
                for k,e in enumerate(index2electrode):
                    for j,f in enumerate(freqs):

                        regi = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
                        assert regi.sum() == 1
                        reg = clean_region(edata[regi].region.values[0])

                        data['bird'].append(bird)
                        data['block'].append(block)
                        data['segment'].append(seg)
                        data['hemi'].append(hemi)
                        data['electrode'].append(e)
                        data['region'].append(reg)
                        data['cell_index'].append(-1)
                        data['site'].append(site)
                        data['freq'].append(int(f))
                        data['r2'].append(eperf[k, j])

            elif decomp == 'spike_rate':
                for ci,e in enumerate(cell_index2electrode):
                    regi = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
                    assert regi.sum() == 1
                    reg = clean_region(edata[regi].region.values[0])

                    data['bird'].append(bird)
                    data['block'].append(block)
                    data['segment'].append(seg)
                    data['hemi'].append(hemi)
                    data['electrode'].append(e)
                    data['region'].append(reg)
                    data['cell_index'].append(ci)
                    data['site'].append(site)
                    data['freq'].append(-1)
                    data['r2'].append(eperf[ci])

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, 'aggregate', 'encoder_perfs_for_glm.csv'), header=True, index=False)


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_Site4_Call1_L_full_psds.h5')
    lags = hf.attrs['lags']
    freqs = hf.attrs['freqs']
    hf.close()
    nlags = len(lags)

    return freqs,lags


def get_encoder_weights_squared(agg, decomp, data_dir='/auto/tdrive/mschachter/data'):

    freqs, lags = get_freqs_and_lags()

    i = agg.df.decomp == decomp
    assert i.sum() > 0
    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))
    wdata = {'region': list(), 'freq': list(), 'xindex': list(), 'eperf':list()}
    Wsq = list()

    for (bird, block, seg, hemi), gdf in g:

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        iindex = gdf['iindex'].values[0]

        eperf = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]
        index2electrode = agg.index2electrode[wkey]

        for k, e in enumerate(index2electrode):

            regi = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert regi.sum() == 1
            reg = clean_region(edata[regi].region.values[0])

            for j, f in enumerate(freqs):
                w = eweights[k, j, :]

                wdata['eperf'].append(eperf[k, j])
                wdata['region'].append(reg)
                wdata['freq'].append(int(f))
                wdata['xindex'].append(len(Wsq))

                w2 = w**2
                w2 /= w2.sum()
                Wsq.append(w2)

    wdf = pd.DataFrame(wdata)
    Wsq = np.array(Wsq)

    # compute the average encoder weights by frequency
    r2_thresh = 0.05
    Wsq_by_freq = np.zeros([len(USED_ACOUSTIC_PROPS), len(freqs)])
    for j, f in enumerate(freqs):
        i = (wdf.freq == int(f)) & (wdf.eperf > r2_thresh)
        ii = wdf.xindex[i].values
        Wsq_by_freq[:, j] = Wsq[ii, :].mean(axis=0)

    # compute the average encoder weights by region
    regs = ['L2', 'CMM', 'CML', 'L1', 'L3', 'NCM']
    Wsq_by_reg = np.zeros([len(USED_ACOUSTIC_PROPS), len(regs)])
    for j, reg in enumerate(regs):
        i = wdf.region == reg
        ii = wdf.xindex[i].values
        Wsq_by_reg[:, j] = Wsq[ii, :].mean(axis=0)

    return wdf,Wsq,Wsq_by_freq,Wsq_by_reg


def reorder_by_row_sum(W):
    n = W.shape[0]
    rsum = W.sum(axis=1)
    new_order = list(sorted(zip(range(n), rsum), key=operator.itemgetter(1), reverse=True))
    Wr = np.array([W[k[0], :] for k in new_order])
    print 'Wr.shape=',Wr.shape
    return Wr,[x[0] for x in new_order]


def get_lfp_and_spike_perfs_for_boxplot(agg, lfp_decomp='full_psds'):

    freqs, lags = get_freqs_and_lags()

    i = (agg.df.decomp == 'spike_rate') | (agg.df.decomp == lfp_decomp)
    df = agg.df[i]

    g = df.groupby(['bird', 'block', 'segment', 'hemi', 'decomp'])

    assert isinstance(agg, AcousticEncoderDecoderAggregator)

    pdata = {'r2':list(), 'f':list()}

    for (bird,block,segment,hemi,decomp),gdf in g:
        assert len(gdf) == 1
        wkey = gdf.wkey.values[0]
        index2electrode = agg.index2electrode[wkey]
        cell_index2electrode = agg.cell_index2electrode[wkey]

        eperfs = agg.encoder_perfs[wkey]

        if decomp.endswith('psds'):
            for k,e in enumerate(index2electrode):
                for j,f in enumerate(freqs):
                    pdata['r2'].append(eperfs[k, j])
                    pdata['f'].append(f)
        elif decomp == 'spike_rate':
            for ci,e in enumerate(cell_index2electrode):
                pdata['r2'].append(eperfs[ci])
                pdata['f'].append(-1)

    return pd.DataFrame(pdata)


def plot_lfp_and_spike_perf_boxplot(agg, ax=None, lfp_decomp='full_psds'):

    if ax is None:
        ax = plt.gca()

    df = get_lfp_and_spike_perfs_for_boxplot(agg, lfp_decomp=lfp_decomp)
    freqs, lags = get_freqs_and_lags()

    bp_data = dict()
    for f in freqs:
        i = df.f == f
        bp_data['%d' % f] = df.r2[i].values
    i = df.f == -1
    bp_data['Spike\nRate'] = df.r2[i].values

    group_names = ['%d' % f for f in freqs]
    group_names.append('Spike\nRate')

    group_colors = [COLOR_BLUE_LFP]*len(freqs)
    group_colors.append(COLOR_RED_SPIKE_RATE)

    boxplot_with_colors(bp_data, group_names, ax=ax, group_colors=group_colors, box_alpha=0.95)
    plt.ylim(0, 0.5)
    plt.ylabel('Encoder R2')
    plt.xlabel('Frequency (Hz)')


def plot_avg_psd_encoder_weights(agg, data_dir='/auto/tdrive/mschachter/data', decomp='full_psds'):

    freqs,lags = get_freqs_and_lags()
    bs_agg = AggregateBiosounds.load(os.path.join(data_dir, 'aggregate', 'biosound.h5'))

    wdf,Wsq,Wsq_by_freq,Wsq_by_reg = get_encoder_weights_squared(agg, decomp)

    print 'Wsq_by_freq.shape=',Wsq_by_freq.shape
    Wsq_by_freq,aprop_order = reorder_by_row_sum(Wsq_by_freq)

    figsize = (23, 10)
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(100, 100)

    pwidth = 45

    # plot the performance boxplots by frequency
    freqs = list(sorted(wdf.freq.unique()))
    ax = plt.subplot(gs[:35, :pwidth])
    plot_lfp_and_spike_perf_boxplot(agg, ax, lfp_decomp=decomp)

    # plot the average encoder effect sizes
    ax = plt.subplot(gs[45:100, :pwidth])
    plt.imshow(Wsq_by_freq, origin='upper', interpolation='nearest', aspect='auto', vmin=0, cmap=magma)
    plt.xticks(range(len(freqs)), ['%d' % int(f) for f in freqs])
    aprop_lbls = [ACOUSTIC_PROP_NAMES[USED_ACOUSTIC_PROPS[k]] for k in aprop_order]
    plt.yticks(range(len(aprop_lbls)), aprop_lbls)
    plt.xlabel('LFP Frequency (Hz)')
    plt.colorbar(label='Normalized Encoder Effect')

    fname = os.path.join(get_this_dir(), 'average_encoder_weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_encoder_perfs(agg):

    freqs,lags = get_freqs_and_lags()

    electrodes = [ROSTRAL_CAUDAL_ELECTRODES_LEFT,
                  ROSTRAL_CAUDAL_ELECTRODES_RIGHT,
                  ROSTRAL_CAUDAL_ELECTRODES_LEFT]

    sites = [('GreBlu9508M', 'Site4', 'Call1', 'L'),
             ('WhiWhi4522M', 'Site2', 'Call2', 'R'),
             ('YelBlu6903F', 'Site3', 'Call3', 'L')
             ]

    weights = list()
    for bird,block,segment,hemi in sites:

        i = (agg.df.bird == bird) & (agg.df.block == block) & (agg.df.segment == segment) & (agg.df.hemi == hemi)

        ii = i & (agg.df.decomp == 'self_locked')
        assert ii.sum() == 1
        wkey = agg.df[ii]['wkey'].values[0]
        lfp_eperf = agg.encoder_perfs[wkey]

        lfp_decoder_weights = agg.decoder_weights[wkey]
        print 'lfp_decoder_weights.shape=',lfp_decoder_weights.shape
        lfp_sal_weights = lfp_decoder_weights[:, :, REDUCED_ACOUSTIC_PROPS.index('sal')]
        lfp_q2_weights = lfp_decoder_weights[:, :, REDUCED_ACOUSTIC_PROPS.index('q2')]

        weights.append( (lfp_eperf, lfp_sal_weights, lfp_q2_weights) )

    figsize = (24, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.25, wspace=0.25)
    nrows = 3
    ncols = 3

    for k,(lfp_eperf,lfp_sal_weights,lfp_q2_weights) in enumerate(weights):

        index2electrode = electrodes[k]

        ax = plt.subplot(nrows, ncols, k*ncols + 1)
        plt.imshow(lfp_eperf, interpolation='nearest', aspect='auto', vmin=0, cmap=magma)
        plt.yticks(range(len(index2electrode)), ['%d' % e for e in index2electrode])
        plt.xticks(range(len(freqs)), ['%d' % f for f in freqs], rotation=45)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Electrode')
        plt.colorbar(label='Encoder R2')

        ax = plt.subplot(nrows, ncols, k*ncols + 2)
        absmax = np.abs(lfp_sal_weights).max()
        plt.imshow(lfp_sal_weights, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.yticks(range(len(index2electrode)), ['%d' % e for e in index2electrode])
        plt.xticks(range(len(freqs)), ['%d' % f for f in freqs], rotation=45)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Electrode')
        plt.colorbar(label='Decoder Weight')

        ax = plt.subplot(nrows, ncols, k*ncols + 3)
        absmax = np.abs(lfp_sal_weights).max()
        plt.imshow(lfp_q2_weights, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.yticks(range(len(index2electrode)), ['%d' % e for e in index2electrode])
        plt.xticks(range(len(freqs)), ['%d' % f for f in freqs], rotation=45)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Electrode')
        plt.colorbar(label='Decoder Weight')

    fname = os.path.join(get_this_dir(), 'encoder_weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_all_encoder_perfs_and_decoder_weights(agg, aprops=('sal', 'q2', 'maxAmp', 'meantime', 'entropytime')):

    freqs,lags = get_freqs_and_lags()

    font = {'family':'normal', 'weight':'bold', 'size':10}
    plt.matplotlib.rc('font', **font)

    plist = list()

    for (bird,block,segment,hemi),gdf in agg.df.groupby(['bird', 'block', 'segment', 'hemi']):

        bstr = '%s_%s_%s_%s' % (bird,hemi,block,segment)
        ii = (gdf.decomp == 'self_locked')
        assert ii.sum() == 1
        wkey = gdf[ii]['wkey'].values[0]

        lfp_eperf = agg.encoder_perfs[wkey]

        plist.append({'type':'encoder', 'X':lfp_eperf, 'title':bstr})

        lfp_decoder_weights = agg.decoder_weights[wkey]
        lfp_decoder_perfs = agg.decoder_perfs[wkey]

        for k,aprop in enumerate(aprops):
            ai = USED_ACOUSTIC_PROPS.index(aprop)
            lfp_weights = lfp_decoder_weights[:, :, ai]
            dstr = '%0.2f (%s)' % (lfp_decoder_perfs[ai], aprop)
            plist.append({'type':'decoder', 'X':lfp_weights, 'title':dstr})

    def _plot_X(_pdata, _ax):
        plt.sca(_ax)
        _X = _pdata['X']
        if _pdata['type'] == 'decoder':
            _absmax = np.abs(_X).max()
            _vmin = -_absmax
            _vmax = _absmax
            _cmap = plt.cm.seismic
        else:
            _vmin = 0.
            _vmax = 0.35
            _cmap = magma

        plt.imshow(_X, interpolation='nearest', aspect='auto', cmap=_cmap, vmin=_vmin, vmax=_vmax)
        plt.title(_pdata['title'])
        plt.xticks([])
        plt.yticks([])

    multi_plot(plist, _plot_X, nrows=5, ncols=6, figsize=(23, 13))
    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'acoustic_encoder_decoder.h5')
    agg = AcousticEncoderDecoderAggregator.load(agg_file)

    # ###### figure with encoder effects per frequency
    plot_avg_psd_encoder_weights(agg, decomp='onewin_psds')

    # ###### figure with encoder effects per lag
    # plot_avg_pairwise_encoder_weights(agg)

    # ###### write a csv file out for analysis in R
    # export_psd_encoder_datasets_for_glm(agg)

    # draw_encoder_perfs(agg)
    # draw_all_encoder_perfs_and_decoder_weights(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()
