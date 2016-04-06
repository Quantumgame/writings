import os
from copy import deepcopy

import h5py
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from lasp.plots import custom_legend

from DecodingRhythms.utils import set_font, get_this_dir, clean_region
from lasp.colormaps import magma

from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import REDUCED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, \
    ACOUSTIC_FEATURE_COLORS


def export_decoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    freqs = hf.attrs['freqs']
    hf.close()

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'decomp':list(), 'site':list(),
            'aprop':list(), 'r2':list()}

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi', 'decomp'])
    for (bird,block,seg,hemi,decomp),gdf in g:

        if decomp not in ['self_locked', 'self_spike_rate', 'self+cross_locked']:
            continue

        site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        iindex = gdf['iindex'].values[0]

        dperf = agg.decoder_perfs[wkey]

        for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):

            r2 = dperf[k]

            data['bird'].append(bird)
            data['block'].append(block)
            data['segment'].append(seg)
            data['hemi'].append(hemi)
            data['decomp'].append(decomp)
            data['site'].append(site)
            data['aprop'].append(aprop)
            data['r2'].append(r2)

    df = pd.DataFrame(data)

    print 'decomps=',df.decomp.unique()
    df.to_csv(os.path.join(data_dir, 'aggregate', 'decoder_perfs_for_glm.csv'), header=True, index=False)


def export_pairwise_encoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    lags = hf.attrs['lags']
    hf.close()

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
        iindex = gdf['iindex'].values[0]
        index2electrode = agg.index2electrode[iindex]

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

                    for ai,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
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


def export_psd_encoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    freqs = hf.attrs['freqs']
    hf.close()

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(), 'region':list(), 'site':list(),
            'freq':list(), 'r2':list()}

    weight_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(), 'region':list(), 'site':list(),
                   'freq':list(), 'aprop':list(), 'w':list()}

    # store the decoder weights for each acoustic property as well
    for aprop in REDUCED_ACOUSTIC_PROPS:
        data['weight_%s' % aprop] = list()

    decomp = 'self_locked'
    i = agg.df.decomp == decomp

    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird,block,seg,hemi),gdf in g:

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        iindex = gdf['iindex'].values[0]

        eperf = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]
        # normalize weights!
        eweights /= np.abs(eweights).max()
        index2electrode = agg.index2electrode[iindex]

        site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

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
                data['site'].append(site)
                data['freq'].append(int(f))
                data['r2'].append(eperf[k, j])

                for ai,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
                    keyname = 'weight_%s' % aprop
                    w = eweights[k, j, ai]
                    data[keyname].append(w)
                    
                    weight_data['bird'].append(bird)
                    weight_data['block'].append(block)
                    weight_data['segment'].append(seg)
                    weight_data['hemi'].append(hemi)
                    weight_data['electrode'].append(e)
                    weight_data['region'].append(clean_region(reg))
                    weight_data['site'].append(site)
                    weight_data['freq'].append(int(f))
                    weight_data['aprop'].append(aprop)
                    weight_data['w'].append(w)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, 'aggregate', 'encoder_perfs_for_glm.csv'), header=True, index=False)

    wdf = pd.DataFrame(weight_data)
    wdf.to_csv(os.path.join(data_dir, 'aggregate', 'encoder_weights_for_glm.csv'), header=True, index=False)


def read_pairwise_encoder_weights_weights(data_dir='/auto/tdrive/mschachter/data'):

    fname = os.path.join(get_this_dir(), 'pairwise_encoder_weights_glm_weights.txt')

    data = {'lag':list(), 'aprop':list(), 'w':list(), 'p':list()}

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    lags = hf.attrs['lags']
    hf.close()
    nlags = len(lags)

    f = open(fname, 'r')
    for ln in f.readlines():
        if len(ln.strip()) == 0:
            continue

        x = ln.strip().split()
        # print 'x=',x

        aprop,lag = x[0].split(':')
        aprop = aprop[5:]
        lag = int(lag[3:])

        if x[1] == 'NA':
            w = 0.
            p = 1.
        else:
            w = float(x[1])
            p = float(x[4])

        data['lag'].append(lag)
        data['aprop'].append(aprop)
        data['w'].append(w)
        data['p'].append(p)

    df = pd.DataFrame(data)

    lags = lags[np.abs(lags) < 52]
    w_mat = np.zeros([len(REDUCED_ACOUSTIC_PROPS), len(lags)])

    for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
        for j,l in enumerate(lags):
            i = (df.lag == int(l)) & (df.aprop == aprop)
            p = df[i].p.values[0]
            w = df[i].w.values[0]
            if p < 0.05:
                w_mat[k, j] = w

    return lags,w_mat


def read_pairwise_encoder_perfs_weights(data_dir='/auto/tdrive/mschachter/data'):

    fname = os.path.join(get_this_dir(), 'pairwise_encoder_perfs_glm_weights.txt')

    data = {'lag':list(), 'region1':list(), 'region2':list(), 'w':list(), 'p':list()}

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    lags = hf.attrs['lags']
    hf.close()
    nlags = len(lags)

    f = open(fname, 'r')
    for ln in f.readlines():
        if len(ln.strip()) == 0:
            continue

        x = ln.strip().split()
        # print 'x=',x

        reg_or_lag = x[0]
        lag = None
        reg1 = None
        reg2 = None
        
        if reg_or_lag.startswith('lag'):
            lag = int(reg_or_lag[3:])
        if reg_or_lag.startswith('region'):
            regs = reg_or_lag[7:]
            reg1,reg2 = regs.split('->')

        w = float(x[1])
        p = float(x[4])

        data['lag'].append(lag)
        data['region1'].append(reg1)
        data['region2'].append(reg2)
        data['w'].append(w)
        data['p'].append(p)

    df = pd.DataFrame(data)

    lags_plotted = lags[np.abs(lags) < 52]
    i = ~np.isnan(df.lag) & (df.lag < 52)
    lag_list = [(lag, w, p) for lag,w,p in zip(df[i].lag.values, df[i].w.values, df[i].p.values)]
    lag_list.sort(key=operator.itemgetter(0))
    lag_list = np.array(lag_list)

    # zero out weights that are not statistically significant
    nss = lag_list[:, -1] > 0.05
    lag_list[nss, 1] = 0.

    lag_weights = np.array([x[1] for x in lag_list])

    regs = ['L2', 'CMM', 'CML', 'L1', 'L3', 'NCM']
    reg_weights = np.zeros([len(regs), len(regs)])

    for k,r1 in enumerate(regs):
        for j,r2 in enumerate(regs):

            i = (df.region1 == r1) & (df.region2 == r2)
            if i.sum() == 0:
                print 'Missing connection: (%s,%s)i.sum()=%d' % (r1, r2, i.sum())
                continue

            w = df[i].w.values[0]
            p = df[i].p.values[0]

            if p < 0.05:
                reg_weights[k, j] = w

    lags_w,w_mat = read_pairwise_encoder_weights_weights()

    figsize = (23, 10)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.90, left=0.10, hspace=0.25, wspace=0.25)

    gs = plt.GridSpec(100, 100)

    ax = plt.subplot(gs[:30, :32])
    plt.plot(lags_plotted, lag_weights, 'k-', alpha=0.7, linewidth=7.0)
    plt.axis('tight')
    plt.ylim(0, 0.10)
    plt.ylabel('Avg. Encoder R2 Contrib')
    plt.xlabel('Lag (ms)')

    ax = plt.subplot(gs[40:, :40])
    absmax = np.abs(w_mat).max()
    plt.imshow(w_mat, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=0, cmap=plt.cm.afmhot,
               origin='lower', extent=(lags_w.min(), lags_w.max(), 0, len(REDUCED_ACOUSTIC_PROPS)))
    xlbls = np.array([-40, -20, 0, 20, 40])
    plt.xticks(xlbls, ['%d' % x for x in xlbls])
    plt.xlabel('Lag (ms)')
    plt.yticks(np.arange(len(REDUCED_ACOUSTIC_PROPS))+0.5, REDUCED_ACOUSTIC_PROPS)
    plt.colorbar(label='Avg Weight Contrib')
    plt.axis('tight')

    ax = plt.subplot(gs[:, 45:])
    absmax = np.abs(reg_weights).max()
    print 'reg_weights='
    print reg_weights
    plt.imshow(reg_weights, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic, origin='lower')
    plt.xticks(range(len(regs)), regs)
    plt.yticks(range(len(regs)), regs)
    plt.colorbar(label='Avg Encoder R2 Contrib')

    fname = os.path.join(get_this_dir(), 'pairwise_encoder_perf+weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def read_encoder_weights_weights(data_dir='/auto/tdrive/mschachter/data'):

    fname = os.path.join(get_this_dir(), 'encoder_weights_glm_weights.txt')

    data = {'aprop':list(), 'freq':list(), 'region':list(), 'w':list(), 'p':list()}

    f = open(fname, 'r')
    for ln in f.readlines():
        if len(ln.strip()) == 0:
            continue

        x = ln.strip().split()
        print 'x=',x

        aprop,reg_or_freq = x[0].split(':')
        aprop = aprop[5:]
        freq = None
        reg = None
        if reg_or_freq.startswith('freq'):
            freq = int(reg_or_freq[4:])
        if reg_or_freq.startswith('region'):
            reg = reg_or_freq[6:]

        w = float(x[1])
        p = float(x[4])

        data['aprop'].append(aprop)
        data['freq'].append(freq)
        data['region'].append(reg)
        data['w'].append(w)
        data['p'].append(p)

    df = pd.DataFrame(data)

    freqs = sorted(df.freq.unique())
    if np.isnan(freqs).sum() > 0:
        freqs = freqs[:-1]

    regs = ['L2', 'CMM', 'CML', 'L1', 'L3', 'NCM']
    aprops = REDUCED_ACOUSTIC_PROPS

    w_by_freq = np.zeros([len(aprops), len(freqs)])
    for k,aprop in enumerate(aprops):
        for j,f in enumerate(freqs):
            i = (df.freq == f) & (df.aprop == aprop)
            if i.sum() == 0:
                continue
            assert i.sum() == 1, "i.sum()=%d" % i.sum()
            p =  df[i].p.values[0]
            if p < 0.05:
                w_by_freq[k, j] = df[i].w.values[0]

    w_by_reg = np.zeros([len(aprops), len(regs)])
    for k,aprop in enumerate(aprops):
        for j,r in enumerate(regs):
            i = (df.region == r) & (df.aprop == aprop)
            if i.sum() == 0:
                continue
            assert i.sum() == 1
            p =  df[i].p.values[0]
            if p < 0.05:
                w_by_reg[k, j] = df[i].w.values[0]

    figsize = (23, 10)
    fig = plt.figure(figsize=figsize)

    gs = plt.GridSpec(1, 100)

    ax = plt.subplot(gs[0, :50])
    absmax = np.abs(w_by_freq).max()
    plt.imshow(w_by_freq, origin='upper', interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
    plt.xticks(range(len(freqs)), ['%d' % int(f) for f in freqs])
    plt.yticks(range(len(aprops)), aprops)
    plt.xlabel('Frequency (Hz)')
    plt.colorbar(label='Encoder Weight Contribution')

    ax = plt.subplot(gs[0, 70:])
    absmax = np.abs(w_by_reg).max()
    plt.imshow(w_by_reg, origin='upper', interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
    plt.xticks(range(len(regs)), ['%s' % r for r in regs])
    plt.yticks(range(len(aprops)), aprops)
    plt.xlabel('Region')
    plt.colorbar(label='Encoder Weight Contribution')

    fname = os.path.join(get_this_dir(), 'average_encoder_weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

def draw_encoder_perfs(agg):

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    freqs = hf.attrs['freqs']
    hf.close()

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


def draw_encoder_weights(agg):

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    freqs = hf.attrs['freqs']
    hf.close()

    aprops = REDUCED_ACOUSTIC_PROPS

    w_by_decomp = dict()

    for decomp in ['self_locked', 'self_spike_psd']:
        all_weights = list()
        i = agg.df.decomp == decomp
        g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
        for (bird,block,esgment,hemi),gdf in g:
            assert len(gdf) == 1
            wkey = gdf.wkey.values[0]
            perfs = agg.encoder_perfs[wkey]
            perfs /= perfs.max()

            W = np.abs(agg.encoder_weights[wkey])
            for k,aprop in enumerate(aprops):
                W[:, :, k] *= perfs

            Wmean = W.mean(axis=0)
            all_weights.append(Wmean)

        all_weights = np.array(all_weights)
        w_by_decomp[decomp] = all_weights.mean(axis=0)

    aprops_to_display = ['sal', 'maxAmp', 'meanspect', 'entropyspect', 'q2', 'entropytime']

    figsize = (24, 7)
    fig = plt.figure(figsize=figsize)
    clrs = [ACOUSTIC_FEATURE_COLORS[aprop] for aprop in aprops_to_display]

    ax = plt.subplot(1, 2, 1)
    for aprop in aprops_to_display:
        m = aprops.index(aprop)
        plt.plot(freqs, w_by_decomp['self_locked'][:, m], '-', c=ACOUSTIC_FEATURE_COLORS[aprop], linewidth=7.0, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Weight Magnitude')
    plt.axis('tight')
    plt.title('Weight Importance (LFP PSD)')
    leg = custom_legend(clrs, aprops_to_display)
    plt.legend(handles=leg)

    ax = plt.subplot(1, 2, 2)
    for aprop in aprops_to_display:
        m = aprops.index(aprop)
        plt.plot(freqs, w_by_decomp['self_spike_psd'][:, m], '-', c=ACOUSTIC_FEATURE_COLORS[aprop], linewidth=7.0, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Weight Magnitude')
    plt.axis('tight')
    plt.title('Weight Importance (Spike PSD)')
    leg = custom_legend(clrs, aprops_to_display)
    plt.legend(handles=leg)

    fname = os.path.join(get_this_dir(), 'encoder_weights_by_freq.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)

    # export_encoder_datasets_for_glm(agg)
    # export_decoder_datasets_for_glm(agg)
    # export_pairwise_encoder_datasets_for_glm(agg)

    # read_encoder_weights_weights()
    read_pairwise_encoder_perfs_weights()

    # draw_encoder_perfs(agg)
    # draw_encoder_weights(agg)

    """
    for (bird,block,segment,hemi,decomp),gdf in agg.df.groupby(['bird', 'block', 'segment','hemi','decomp']):

        assert len(gdf) == 1

        if decomp in ['self_spike_rate', 'self+cross_locked']:
            continue

        # get segment/decomp props
        wkey = gdf.wkey.values[0]
        iindex = gdf.iindex.values[0]

        # get encoder performances
        encoder_perfs = agg.encoder_perfs[wkey]

        # get decoder weights
        decoder_weights = agg.decoder_weights[wkey]
        decoder_perfs = agg.decoder_perfs[wkey]

        fig = plt.figure(figsize=(24, 13))
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.15, wspace=0.15)
        ncols = 4
        nrows = 3

        plt.subplot(nrows, ncols, 1)
        plt.imshow(encoder_perfs, interpolation='nearest', aspect='auto', vmin=0, cmap=magma)
        plt.colorbar()
        plt.xticks()
        plt.title('Encoder Perf')

        for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
            dW = decoder_weights[:, :, k]
            plt.subplot(nrows, ncols, k+2)
            absmax = np.abs(dW).max()
            plt.imshow(dW, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
            plt.colorbar()
            plt.title('%s decoder: %0.2f' % (aprop, decoder_perfs[k]))

        seg_fname = '%s_%s_%s_%s_%s' % (bird, block, segment, hemi, decomp)
        plt.suptitle(seg_fname)
        fname = os.path.join(fig_dir, '%s.png' % seg_fname)

        plt.savefig(fname)
        plt.close('all')
    """

if __name__ == '__main__':
    set_font()
    draw_figures()
