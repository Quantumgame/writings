import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from utils import set_font, get_this_dir, get_freqs

from zeebeez.aggregators.acoustic_decoders import AggregateAcousticDecoder


def get_adata(df, band, decomp='locked', order='self'):
    i = (df.order == order) & (df.decomp == decomp) & (df.band == band) & ~df.exfreq
    aprops = df.aprop.unique()

    # measure the mean and sd cc for each acoustic property
    adata = {'aprop':list(), 'cc_mean':list(), 'cc_std':list(), 'r2_mean':list(), 'r2_std':list()}
    for aprop in aprops:

        ii = i & (df.aprop == aprop)
        ccs = df[ii]['cc'].values
        r2s = df[ii]['r2'].values

        adata['aprop'].append(aprop)
        adata['cc_mean'].append(ccs.mean())
        adata['cc_std'].append(ccs.std(ddof=1))
        adata['r2_mean'].append(r2s.mean())
        adata['r2_std'].append(r2s.std(ddof=1))

    adf = pd.DataFrame(adata)
    return adf.sort('r2_mean', ascending=False)


def get_prop_perf_by_freq(df, aprop, nbands=12, perf='r2'):

    pp = list()

    for b in range(nbands):
        adata = get_adata(df, b+1)
        i = adata.aprop == aprop
        assert i.sum() == 1
        pp.append(adata[i]['%s_mean' % perf].values[0])
    pp = np.array(pp)

    return pp


def draw_perf_plots(agg, df, aprops, perf='cc'):

    orders = ['self', 'cross', 'self+cross']
    decomps = ['locked', 'nonlocked', 'total']

    freqs = agg.freqs[0]
    lags = agg.lags[0]
    nelectrodes = len(agg.index2electrode[0])

    perfs = dict()
    nsamps = None
    for o in orders:
        for d in decomps:
            for a in aprops:
                i = (df.order == o) & (df.decomp == d) & (df.band == 0) & ~df.exfreq & (df.aprop == a)
                if nsamps is None:
                    nsamps = i.sum()
                assert i.sum() == nsamps

                if perf == 'aic':
                    # compute the AIC
                    nparams = 0
                    if 'self' in o:
                        nparams += len(freqs)*nelectrodes
                    if 'cross' in o:
                        nparams += len(lags)*((nelectrodes**2 - nelectrodes) / 2)
                    if d == 'total':
                        nparams *= 2

                    ll = -df['likelihood'][i].values
                    print '%s/%s/%s: nparams=%d' % (o, d, a, nparams)
                    aic = 2*(nparams - ll)
                    vals = aic / 1e3

                else:
                    vals = df[perf][i].values

                key = (o, d, a)
                # perfs[key] = df.rmse[i].values
                perfs[key] = vals

    fig = plt.figure(figsize=(24, 13))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.30, wspace=0.20)

    nrows = 3
    ncols = 3
    for k,a in enumerate(aprops):

        nonlocked_perf = np.array([perfs[(o, 'nonlocked', a)].mean() for o in orders])
        nonlocked_perf_std = np.array([perfs[(o, 'nonlocked', a)].std(ddof=1) for o in orders])

        locked_perf = np.array([perfs[(o, 'locked', a)].mean() for o in orders])
        locked_perf_std = np.array([perfs[(o, 'locked', a)].std(ddof=1) for o in orders])

        total_perf = np.array([perfs[(o, 'total', a)].mean() for o in orders])
        total_perf_std = np.array([perfs[(o, 'total', a)].std(ddof=1) for o in orders])

        ax = plt.subplot(nrows, ncols, k+1)
        x = np.arange(3) + 1
        plt.bar(x, locked_perf, width=0.2, yerr=nonlocked_perf_std/np.sqrt(nsamps), facecolor='k', ecolor='k')
        plt.bar(x+0.25, nonlocked_perf, width=0.2, yerr=locked_perf_std/np.sqrt(nsamps), facecolor='w', alpha=0.75, ecolor='k')
        plt.bar(x+0.5, total_perf, width=0.2, yerr=total_perf_std/np.sqrt(nsamps), facecolor='#C0C0C0', alpha=0.75, ecolor='k')
        plt.axis('tight')
        plt.xlim(0.65, 4)
        plt.xticks(x+0.35, orders)
        if perf in ['cc', 'r2']:
            plt.ylim(0, 1)
        plt.legend(['Trial-avg', 'Mean-sub', 'Both'], fontsize='x-small', loc=4)
        plt.title(a)
        plt.ylabel(perf)

    fname = os.path.join(get_this_dir(), 'model_%s.svg' % perf)
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_freq_plots(agg, df, aprops):

    freqs = agg.freqs[0]
    bands = sorted(agg.df.band.unique())

    print 'band/freq:'
    for b,f in zip(bands, freqs):
        print '%d   %0.2f Hz' % (b+1, f)

    clrs = ['#005E98', '#57BEE4', '#DA0030', '#F36078', '#009C4F', '#966A56', '#7F4C39', '#5C3125', 'k']

    fig = plt.figure(figsize=(24, 8))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.30, wspace=0.20)

    for k,aprop in enumerate(aprops):
        pp = get_prop_perf_by_freq(df, aprop)
        plt.plot(freqs, pp, '-', c=clrs[k], linewidth=7.0, alpha=0.75)
    plt.legend(aprops, fontsize='medium')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean R2 Across Sites')
    plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'perf_by_freq.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_weight_plots(agg, df, aprop):

    sample_rate = 381.4697265625
    freqs = get_freqs(sample_rate)
    edata = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/electrode_data.csv')

    i = (df.order == 'self') & (df.decomp == 'locked') & (df.band == 0) & (df.exfreq == False) & (df.aprop == aprop)
    g = df[i].groupby(['bird', 'block', 'segment', 'hemi'])

    for (bird,block,segment,hemi),gdf in g:

        assert len(gdf) == 1

        cc = gdf['cc'][i].values[0]
        index = gdf['index'][i].values[0]
        index2electrode = agg.index2electrode[index]

        wkey = ('self', 'locked', 0, False)
        wi = gdf.windex[i].values[0]
        w = agg.weights[wkey][wi]

        nelectrodes = len(index2electrode)
        nfreqs = len(freqs)

        index2coord = list()
        index2label = list()
        for n,e in enumerate(index2electrode):
            ii = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert ii.sum() == 1
            row = edata[ii]['row'].values[0]
            col = edata[ii]['col'].values[0]
            reg = edata[ii]['region'].values[0]
            index2coord.append((row, col))
            lbl = '%d\n%s' % (e, reg)
            index2label.append(lbl)

        W = w.reshape([nelectrodes, nfreqs])
        nrows = 8
        ncols = 2
        Wgrid = np.zeros([nrows, ncols, nfreqs])

        for n,(row,col) in enumerate(index2coord):
            Wgrid[row, col, :] = W[n, :]

        fig = plt.figure(figsize=(24, 13))
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.30, wspace=0.20)

        nrows = 2
        ncols = 6
        absmax = np.abs(Wgrid).max()
        for k,f in enumerate(freqs):
            W = Wgrid[:, :, k]
            ax = plt.subplot(nrows, ncols, k+1)
            plt.imshow(W, origin='upper', interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax,
                       cmap=plt.cm.seismic, extent=[0, 2, 0, 8])

            for txt,(row,col) in zip(index2label, index2coord):
                plt.text(col + 0.5, (7 - row) + 0.25, txt, horizontalalignment='center', color='k', fontsize=14)

            # plt.colorbar()
            plt.title('%d Hz' % f)
            plt.xticks([])
            plt.yticks([])

        fname = '%s_%s_%s_%s_%s' % (aprop, bird, block, segment, hemi)
        plt.suptitle('%s %0.2f' % (fname, cc))
        fname = os.path.join(get_this_dir(), 'weights_%s.svg' % fname)
        print 'Saving %s' % fname
        plt.savefig(fname, facecolor='w', edgecolor='none')
        plt.close('all')


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'acoustic_decoders.h5')
    agg = AggregateAcousticDecoder.load(agg_file)

    df = agg.df

    aprops = ['meantime', 'stdtime', 'meanspect', 'stdspect', 'sal', 'q1', 'q2', 'q3', 'maxAmp']
    draw_freq_plots(agg, df, aprops)
    # draw_perf_plots(agg, df, aprops)

    # for aprop in aprops:
    #     draw_weight_plots(agg, df, aprop)


if __name__ == '__main__':

    set_font()
    draw_figures()
