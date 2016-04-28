import os
from copy import deepcopy, copy

import h5py
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from lasp.plots import custom_legend, grouped_boxplot, multi_plot

from DecodingRhythms.utils import set_font, get_this_dir, clean_region, COLOR_RED_SPIKE_RATE, COLOR_BLUE_LFP, \
    COLOR_PURPLE_LFP_CROSS
from lasp.colormaps import magma
from zeebeez.aggregators.biosound import AggregateBiosounds

from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import REDUCED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, \
    ACOUSTIC_FEATURE_COLORS, ALL_ACOUSTIC_PROPS


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

                    if reg1 in ['HP', '?'] or reg2 in ['HP', '?']:
                        continue

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

                        if reg1 in ['HP', '?'] or reg2 in ['HP', '?']:
                            continue

                        if abs(lag) > 20:
                            continue

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

    decomp = 'self+cross_locked'
    i = agg.df.decomp == decomp

    W = list()
    wdata = {'lag':list(), 'xindex':list()}

    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird,block,seg,hemi),gdf in g:

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        iindex = gdf['iindex'].values[0]
        index2electrode = agg.index2electrode[iindex]

        eperf = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]
        eweights_whitened = agg.encoder_weights_whitened[wkey]
        # print("eperf.shape=" + str(eperf.shape))
        # print("eweights.shape=" + str(eweights.shape))
        # normalize weights!
        # eweights /= np.abs(eweights).max()

        # site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

        lag_i = np.abs(lags) < 20

        for k,e1 in enumerate(index2electrode):
            for j in range(k):

                for l,lag in enumerate(lags[lag_i]):

                    w_white = eweights_whitened[k, j, l, :]

                    # first compute the effect size for each weight in the whitened space
                    esize = w_white**2

                    # normalize effect size by dividing by sum
                    esize /= esize.sum()

                    # adjust each weight in proportion to it's effect size
                    w_white *= esize

                    # inverse transform the whitened and rescaled weights
                    w = bs_agg.pca.inverse_transform(w_white)
                    w = eweights[k, j, l, :]

                    wdata['lag'].append(int(lag))
                    wdata['xindex'].append(len(W))
                    W.append(w)

    wdf = pd.DataFrame(wdata)
    W = np.array(W)
    W[np.isnan(W)] = 0.
    W = W**2

    W_by_lag = np.zeros([len(REDUCED_ACOUSTIC_PROPS), lag_i.sum()])
    for l,lag in enumerate(lags[lag_i]):
        i = wdf.lag == int(lag)
        ii = wdf.xindex[i].values
        W_by_lag[:, l] = W[ii, :].mean(axis=0)

    plot = True
    if plot:
        plt.figure()
        # ax = plt.subplot(1, 3, 1)
        absmax = np.abs(W_by_lag).max()
        plt.imshow(W_by_lag, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-absmax, vmax=absmax)
        plt.colorbar()
        plt.title('Mean Weights')
        plt.yticks(np.arange(len(REDUCED_ACOUSTIC_PROPS)), REDUCED_ACOUSTIC_PROPS)
        plt.xticks(np.arange(lag_i.sum()), ['%d' % x for x in lags[lag_i]])

        plt.show()

    return lags[lag_i],W_by_lag,None


def export_psd_encoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs,lags = get_freqs_and_lags()

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

    freqs,lags = get_freqs_and_lags()
    nlags = len(lags)

    f = open(fname, 'r')
    for ln in f.readlines():
        if len(ln.strip()) == 0:
            continue

        x = ln.strip().split()
        # print 'x=',x

        if x[0].startswith('regions'):
            continue

        try:
            aprop,lag = x[0].split(':')
        except ValueError:
            print x[0]
            raise
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


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    lags = hf.attrs['lags']
    freqs = hf.attrs['freqs']
    hf.close()
    nlags = len(lags)

    return freqs,lags


def read_pairwise_encoder_perfs_weights(agg, data_dir='/auto/tdrive/mschachter/data'):

    fname = os.path.join(get_this_dir(), 'pairwise_encoder_perfs_glm_weights.txt')

    data = {'lag':list(), 'region1':list(), 'region2':list(), 'w':list(), 'p':list()}

    freqs,lags = get_freqs_and_lags()

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

    lags_w,w_mat,w_mat_std = plot_avg_pairwise_encoder_weights(agg)
    print("wmat.shape=" + str(w_mat.shape))
    print("lags_w.shape=" + str(lags_w.shape))

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
    absmax = np.abs(reg_weights).max()
    print 'reg_weights='
    print reg_weights
    plt.imshow(reg_weights, interpolation='nearest', aspect='auto', cmap=magma, origin='lower')
    plt.xticks(range(len(regs)), regs)
    plt.yticks(range(len(regs)), regs)
    plt.colorbar(label='Avg Encoder R2 Contrib')

    ax = plt.subplot(gs[:, 55:])
    absmax = np.abs(w_mat).max()
    plt.imshow(w_mat, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic,
               origin='lower')
    plt.xticks(np.arange(len(lags_w)), ['%d' % x for x in lags_w])
    plt.xlabel('Lag (ms)')
    plt.yticks(np.arange(len(REDUCED_ACOUSTIC_PROPS)), REDUCED_ACOUSTIC_PROPS)
    plt.colorbar(label='Mean Weight')
    plt.axis('tight')

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

def plot_avg_psd_encoder_weights(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs,lags = get_freqs_and_lags()
    bs_agg = AggregateBiosounds.load(os.path.join(data_dir, 'aggregate', 'biosound.h5'))

    # freqs = freqs[freqs < 70]

    i = agg.df.decomp == 'self_locked'
    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))
    wdata = {'region':list(), 'freq':list(), 'xindex':list()}
    W = list()
    Wadj = list()
    Wwhite = list()

    for (bird,block,seg,hemi),gdf in g:

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        iindex = gdf['iindex'].values[0]

        eperf = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]
        eweights_whitened = agg.encoder_weights_whitened[wkey]
        index2electrode = agg.index2electrode[iindex]

        for k,e in enumerate(index2electrode):

            regi = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert regi.sum() == 1
            reg = clean_region(edata[regi].region.values[0])

            for j,f in enumerate(freqs):
                w_white = eweights_whitened[k, j, :]
                w = bs_agg.pca.inverse_transform(w_white)

                # first compute the effect size for each weight in the whitened space
                esize = w_white**2

                # normalize effect size by dividing by sum
                esize /= esize.sum()

                # adjust each weight in proportion to it's effect size
                w_white_cpy = w_white*esize

                # inverse transform the whitened and rescaled weights
                w_adj = bs_agg.pca.inverse_transform(w_white_cpy)


                wdata['region'].append(reg)
                wdata['freq'].append(int(f))
                wdata['xindex'].append(len(Wadj))
                W.append(w)
                Wadj.append(w_adj)
                Wwhite.append(w_white)

    wdf = pd.DataFrame(wdata)

    W = np.array(W)
    Wsq = W**2

    # bs_agg.pca.components_ = np.abs(bs_agg.pca.components_)
    # bs_agg.pca.mean_ = 0.

    # print("# of nans: %d" % np.sum(np.isnan(W.ravel())))

    # compute the average encoder weights by frequency
    Wmean_by_freq = np.zeros([len(REDUCED_ACOUSTIC_PROPS), len(freqs)])
    Wstd_by_freq = np.zeros([len(REDUCED_ACOUSTIC_PROPS), len(freqs)])

    for j,f in enumerate(freqs):
        i = wdf.freq == int(f)
        ii = wdf.xindex[i].values

        Wmean_by_freq[:, j] = Wsq[ii, :].mean(axis=0)
        Wstd_by_freq[:, j] = W[ii, :].std(axis=0, ddof=1)

    # compute the average encoder weights by region
    regs = ['L2', 'CMM', 'CML', 'L1', 'L3', 'NCM']
    Wmean_by_reg = np.zeros([len(REDUCED_ACOUSTIC_PROPS), len(regs)])
    for j,reg in enumerate(regs):
        i = wdf.region == reg
        ii = wdf.xindex[i].values
        Wmean_by_reg[:, j] = Wsq[ii, :].mean(axis=0)

    figsize = (23, 10)
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 100)

    absmax = max(np.abs(Wmean_by_freq).max(), np.abs(Wmean_by_reg).max())

    ax = plt.subplot(gs[0, :50])
    plt.imshow(Wmean_by_freq, origin='upper', interpolation='nearest', aspect='auto', vmin=0, vmax=absmax, cmap=magma)
    plt.xticks(range(len(freqs)), ['%d' % int(f) for f in freqs])
    plt.yticks(range(len(REDUCED_ACOUSTIC_PROPS)), REDUCED_ACOUSTIC_PROPS)
    plt.xlabel('Frequency (Hz)')
    plt.colorbar(label='Mean Encoder Effect')

    ax = plt.subplot(gs[0, 65:])
    plt.imshow(Wmean_by_reg, origin='upper', interpolation='nearest', aspect='auto', vmin=0, vmax=absmax, cmap=magma)
    plt.xticks(range(len(regs)), regs)
    plt.yticks(range(len(REDUCED_ACOUSTIC_PROPS)), REDUCED_ACOUSTIC_PROPS)
    plt.xlabel('Region')
    plt.colorbar(label='Mean Encoder Effect')

    fname = os.path.join(get_this_dir(), 'average_encoder_weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


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


def draw_decoder_perf_boxplots(data_dir='/auto/tdrive/mschachter/data'):

    aprops_to_display = list(REDUCED_ACOUSTIC_PROPS)
    aprops_to_display.remove('fund')
    aprops_to_display.remove('voice2percent')

    decomps = ['self_spike_rate', 'self_locked', 'self+cross_locked']
    sub_names = ['Spike Rate', 'LFP PSD', 'LFP Pairwise']
    sub_clrs = [COLOR_RED_SPIKE_RATE, COLOR_BLUE_LFP, COLOR_PURPLE_LFP_CROSS]

    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'decoder_perfs_for_glm.csv'))
    bp_data = dict()

    for aprop in aprops_to_display:

        bd = list()
        for decomp in decomps:
            i = (df_me.decomp == decomp) & (df_me.aprop == aprop)
            perfs = df_me.r2[i].values
            bd.append(perfs)
            print("aprop=%s, decomp=%s" % (aprop, decomp))
            print(perfs)

        bp_data[aprop] = bd

    figsize = (16, 5.5)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    grouped_boxplot(bp_data, group_names=aprops_to_display, subgroup_names=sub_names,
                    subgroup_colors=sub_clrs, box_spacing=1.)

    plt.xlabel('Acoustic Feature')
    plt.ylabel('Decoder R2')

    fname = os.path.join(get_this_dir(), 'decoder_perf_boxplots.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

def draw_encoder_weights(agg):

    freqs,lags = get_freqs_and_lags()

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
            ai = ALL_ACOUSTIC_PROPS.index(aprop)
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

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)

    # draw_all_encoder_perfs_and_decoder_weights(agg)
    # plot_avg_psd_encoder_weights(agg)
    # plot_avg_pairwise_encoder_weights(agg)

    # draw_decoder_perf_boxplots()

    # export_psd_encoder_datasets_for_glm(agg)
    # export_decoder_datasets_for_glm(agg)
    # export_pairwise_encoder_datasets_for_glm(agg)

    # read_encoder_weights_weights()
    # read_pairwise_encoder_perfs_weights(agg)
    # get_avg_encoder_weights(agg)

    # get_avg_encoder_weights(agg)

    # draw_encoder_perfs(agg)
    # draw_encoder_weights(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()
