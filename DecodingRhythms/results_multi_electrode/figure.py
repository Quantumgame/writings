import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from utils import set_font, get_this_dir

from zeebeez.aggregators.pairwise_decoders_multi import AggregatePairwiseDecoder
from zeebeez.utils import DECODER_CALL_TYPES, CALL_TYPES, CALL_TYPE_SHORT_NAMES


def plot_pcc_hist(agg, df):

    plt.figure()
    plt.hist(df.pcc, bins=20, color='r')
    plt.xlabel('PCC')
    plt.axis('tight')


def plot_perf_bars(agg, df, perf='pcc'):

    g = df.groupby(['order', 'decomp'])

    aic_mean = dict()
    aic_std = dict()

    perf_mean = dict()
    perf_std = dict()

    nclasses = len(agg.class_names[0])
    nelectrodes = len(agg.index2electrode[0])
    nfreqs = len(agg.freqs[0])
    nlags = len(agg.lags[0])

    print 'nfreqs=%d, nlags=%d' % (nfreqs, nlags)

    nsamps = 0
    for (otype,decomp),gdf in g:

        nparams = 0
        if 'self' in otype:
            nparams += nfreqs*nelectrodes

        if 'cross' in otype:
            nparams += nlags*((nelectrodes**2 - nelectrodes) / 2)

        if decomp == 'total':
            nparams *= 2

        nparams *= nclasses

        nsamps = len(gdf)
        print '%s/%s, nparams=%d, nsamps=%d' % (otype, decomp, nparams, nsamps)

        ll = -gdf.likelihood
        ll *= gdf.num_samps

        aic = 2*(nparams - ll)

        aic /= 1e3

        aic_mean[(otype, decomp)] = aic.mean()
        aic_std[(otype, decomp)] = aic.std(ddof=1)

        perf_mean[(otype, decomp)] = gdf[perf].mean()
        perf_std[(otype, decomp)] = gdf[perf].std(ddof=1)

    decomps = ['locked', 'nonlocked', 'total']
    orders = ['self', 'cross', 'self+cross']

    nonlocked_means_aic = np.array([aic_mean[(o, 'nonlocked')] for o in orders])
    nonlocked_stds_aic = np.array([aic_std[(o, 'nonlocked')] for o in orders])

    locked_means_aic = np.array([aic_mean[(o, 'locked')] for o in orders])
    locked_stds_aic = np.array([aic_std[(o, 'locked')] for o in orders])

    total_means_aic = np.array([aic_mean[(o, 'total')] for o in orders])
    total_stds_aic = np.array([aic_std[(o, 'total')] for o in orders])
    
    nonlocked_means_perf = np.array([perf_mean[(o, 'nonlocked')] for o in orders])
    nonlocked_stds_perf = np.array([perf_std[(o, 'nonlocked')] for o in orders])

    locked_means_perf = np.array([perf_mean[(o, 'locked')] for o in orders])
    locked_stds_perf = np.array([perf_std[(o, 'locked')] for o in orders])

    total_means_perf = np.array([perf_mean[(o, 'total')] for o in orders])
    total_stds_perf = np.array([perf_std[(o, 'total')] for o in orders])

    # compute mean confusion matrix for self,total
    class_names = agg.class_names[0]
    i = (df.order == 'self+cross') & (df.decomp == 'total')
    indices = df['index'][i]
    Cmats = agg.confusion_matrices[indices]
    Cmean = Cmats.mean(axis=0)
    Cro = reorder_confidence_matrix(Cmean, class_names)
    aic_mean = np.mean(np.diag(Cro))

    decomp_map = {'locked':'Trial-averaged', 'nonlocked':'Mean-subtracted', 'total':'Both'}
    decomp_leg = [decomp_map[d] for d in decomps]

    # make plots
    x = np.arange(3) + 1

    fig = plt.figure(figsize=(24, 6))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.30, wspace=0.20)

    ax = plt.subplot(1, 3, 1)
    plt.bar(x, locked_means_perf, width=0.2, yerr=nonlocked_stds_perf/np.sqrt(nsamps), facecolor='k', ecolor='k')
    plt.bar(x+0.25, nonlocked_means_perf, width=0.2, yerr=locked_stds_perf/np.sqrt(nsamps), facecolor='w', alpha=0.75, ecolor='k')
    plt.bar(x+0.5, total_means_perf, width=0.2, yerr=total_stds_perf/np.sqrt(nsamps), facecolor='#C0C0C0', alpha=0.75, ecolor='k')
    plt.axis('tight')
    plt.xlim(0.65, 4)
    plt.xticks(x+0.35, orders)
    plt.legend(decomp_leg, fontsize='x-small', loc=4)
    plt.title('Multi-electrode Performance')
    plt.ylabel(perf)

    ax = plt.subplot(1, 3, 2)
    plt.bar(x, locked_means_aic, width=0.2, yerr=nonlocked_stds_aic/np.sqrt(nsamps), facecolor='k', ecolor='k')
    plt.bar(x+0.25, nonlocked_means_aic, width=0.2, yerr=locked_stds_aic/np.sqrt(nsamps), facecolor='w', alpha=0.75, ecolor='k')
    plt.bar(x+0.5, total_means_aic, width=0.2, yerr=total_stds_aic/np.sqrt(nsamps), facecolor='#C0C0C0', alpha=0.75, ecolor='k')
    plt.axis('tight')
    plt.xlim(0.65, 4)
    plt.xticks(x+0.35, orders)
    plt.legend(decomp_leg, fontsize='x-small', loc=4)
    plt.title('Model Complexity')
    plt.ylabel('AIC * 1e-3')

    ax = plt.subplot(1, 3, 3)
    plt.imshow(Cro, origin='lower', interpolation='nearest', aspect='auto', vmin=0, vmax=1, cmap=plt.cm.afmhot)

    xtks = [CALL_TYPE_SHORT_NAMES[ct] for ct in DECODER_CALL_TYPES]
    plt.xticks(range(len(DECODER_CALL_TYPES)), xtks)
    plt.yticks(range(len(DECODER_CALL_TYPES)), xtks)
    plt.colorbar(label='PCC')
    plt.title('Mean Confusion (self+cross, total) PCC=%0.2f' % aic_mean)

    fname = os.path.join(get_this_dir(), 'model_goodness.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def reorder_confidence_matrix(C, class_names):
    Cro = np.zeros_like(C)
    cnames = list(class_names)
    for i,cname1 in enumerate(cnames):
        for j,cname2 in enumerate(cnames):
            ii = DECODER_CALL_TYPES.index(cname1)
            jj = DECODER_CALL_TYPES.index(cname2)
            Cro[ii, jj] = C[i, j]

    return Cro


def plot_mean_confusion(agg, df):

    g = df.groupby(['order', 'decomp'])

    cmats = dict()
    pccs = dict()
    for (otype,decomp),gdf in g:
        indices = gdf['index'].values
        Cmats = agg.confusion_matrices[indices]
        cmats[(otype, decomp)] = Cmats.mean(axis=0)
        pccs[(otype, decomp)] = gdf.pcc.mean()

    plt.figure()
    sp = 0
    for otype in ('self', 'cross', 'self+cross'):
        for decomp in (['nonlocked', 'locked', 'total']):
            Cmean = cmats[(otype, decomp)]
            pcc = pccs[(otype, decomp)]

            ax = plt.subplot(3, 3, sp+1)
            plt.imshow(Cmean, cmap=plt.cm.afmhot, vmin=0, vmax=1, interpolation='nearest', aspect='auto', origin='lower')
            plt.xticks(range(len(DECODER_CALL_TYPES)), DECODER_CALL_TYPES)
            plt.yticks(range(len(DECODER_CALL_TYPES)), DECODER_CALL_TYPES)
            plt.colorbar()
            plt.title('%s,%s: %0.2f' % (otype, decomp, pcc))

            sp += 1


def plot_psd_weights(agg, df):

    i = (df['decomp'] == 'locked') & (df['order'] == 'self')
    indices = df['psd_index'][i].values
    psds = agg.psds[('self', 'locked')][indices]

    psds = psds.reshape([psds.shape[0]*psds.shape[1]*psds.shape[2], psds.shape[3]])
    psds = np.abs(psds)

    psd_sum = psds.sum(axis=1)
    zi = np.abs(psd_sum) == 0

    psd_std = psds[~zi, :].std(axis=0, ddof=1)
    psd_mean = psds[~zi, :].mean(axis=0)
    psd_cv = psd_std / psd_mean
    freqs = agg.freqs[0]
    print 'freqs=',freqs

    fig = plt.figure(figsize=(24, 16))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    ax = plt.subplot(2, 3, 1)
    plt.plot(freqs, psd_mean, 'k-', linewidth=2.0)
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean')

    ax = plt.subplot(2, 3, 2)
    plt.plot(freqs, psd_std, 'k-', linewidth=7.0, alpha=0.7)
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Model Abs. Weight SD')
    plt.title('Decoder Weight Variation')

    ax = plt.subplot(2, 3, 3)
    plt.plot(freqs, psd_cv, 'k-', linewidth=2.0)
    plt.axis('tight')
    plt.ylim(-25, 25)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('CV')

    fname = os.path.join(get_this_dir(), 'figs.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_figures():

    agg = AggregatePairwiseDecoder.load('/auto/tdrive/mschachter/data/aggregate/decoders_pairwise_coherence_multi.h5')
    agg.df.to_csv('/auto/tdrive/mschachter/data/aggregate/decoder_coherence_multi.csv', index=False)

    # plot_mean_confusion(agg, agg.df)
    # plot_psd_weights(agg, agg.df)
    plot_perf_bars(agg, agg.df)


if __name__ == '__main__':
    set_font()
    draw_figures()
    plt.show()

