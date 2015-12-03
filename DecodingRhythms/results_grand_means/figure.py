import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from layerz.flda import FLDA
from utils import log_transform, set_font, get_this_dir
from zeebeez.aggregators.pairwise_cf import AggregatePairwiseCF
from zeebeez.utils import DECODER_CALL_TYPES, CALL_TYPE_COLORS


def filter_bad_points_cf(df, exclude_regions=True):

    exclude_i = np.zeros(len(df), dtype='bool')
    exclude_i |= (df.bird == 'BlaBro09xxF') & (df.hemi == 'L')

    if exclude_regions:
        exclude_i |= df.region1 == 'HP'
        exclude_i |= df.region1 == '?'
        exclude_i |= df.region1 == 'L'
        exclude_i |= df.region1.str.contains('-')

        exclude_i |= df.region2 == 'HP'
        exclude_i |= df.region2 == '?'
        exclude_i |= df.region2 == 'L'
        exclude_i |= df.region2.str.contains('-')

    print '%d of %d points excluded, %d points left' % (exclude_i.sum(), len(df), len(df) - exclude_i.sum())

    return df[~exclude_i]


def draw_per_region_means(agg, df):

    call_types = DECODER_CALL_TYPES
    print 'call_types=',call_types

    # get single electrode stats per class
    single_stats = dict()
    for ct in call_types:
        i = (df.electrode1 == df.electrode2) & (df.decomp == 'locked') & (df.stim_type == ct)
        print '# of single electrodes for call type %s: %d' % (ct, i.sum())
        indices = df['index'][i].values
        psds = agg.psds[indices]
        log_transform(psds)
        psd_mean = psds.mean(axis=0)
        psd_std = psds.std(axis=0, ddof=1)
        single_stats[ct] = (psd_mean, psd_std)

    # get coherence stats per class
    coherence_stats = dict()
    for ct in call_types:
        i = (df.electrode1 != df.electrode2) & (df.decomp == 'locked') & (df.stim_type == ct)
        print '# of coherence electrodes for call type %s: %d' % (ct, i.sum())
        indices = df['index'][i].values
        psds = agg.psds[indices]
        log_transform(psds)
        psd_mean = psds.mean(axis=0)
        psd_std = psds.std(axis=0, ddof=1)
        coherence_stats[ct] = (psd_mean, psd_std)    

    fi = agg.freqs > 10

    figsize = (19, 8)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    # plot the spectra
    clrs = CALL_TYPE_COLORS
    for ct in call_types:
        m,sd = single_stats[ct]
        cv = sd / m
        ax = plt.subplot(1, 2, 1)
        plt.plot(agg.freqs[fi], cv[fi], '-', c=clrs[ct], linewidth=3.0, alpha=0.75)
    plt.legend(call_types, loc=2, fontsize='x-small')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coefficient of Variation')
    plt.title('Power Spectra')
    plt.axis('tight')
    
    clrs = CALL_TYPE_COLORS
    for ct in call_types:
        m,sd = coherence_stats[ct]
        cv = sd / m
        ax = plt.subplot(1, 2, 2)
        plt.plot(agg.freqs[fi], cv[fi], '-', c=clrs[ct], linewidth=3.0, alpha=0.75)
    plt.legend(call_types, loc=2, fontsize='x-small')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coefficient of Variation')
    plt.title('Power Spectra')
    plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures(agg):

    i = (agg.df.decomp == 'locked') & (agg.df.electrode1 == agg.df.electrode2)
    df = filter_bad_points_cf(agg.df[i])

    # draw_per_region_means(agg, df)
    # return

    indices = df['index'].values
    psds = agg.psds[indices]
    log_transform(psds)

    freqs = agg.freqs
    print '# of points: %d' % len(df)
    print 'len(freqs)=%d' % len(freqs)
    psd_mean = psds.mean(axis=0)
    psd_std = psds.std(axis=0, ddof=1)
    psd_cv = psd_std / psd_mean

    stim_types = df['stim_type'].values
    index2stype = list(np.unique(stim_types))
    X = deepcopy(psds)
    X -= X.mean(axis=0)
    X /= X.std(axis=0, ddof=1)

    y = np.array([index2stype.index(st) for st in stim_types])

    flda = FLDA()
    flda.fit(X, y)
    print 'disc funcs=',flda.discriminant_functions.shape

    figsize = (24, 16)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    ax = plt.subplot(2, 3, 1)
    plt.plot(freqs, psd_std, 'k-', linewidth=3.0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Standard Deviation')
    plt.title('Power Spectra SD')
    plt.axis('tight')

    ax = plt.subplot(2, 3, 2)
    plt.errorbar(freqs, psd_mean, yerr=psd_std, ecolor='r', color='k', elinewidth=3.0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coefficient of Variation')
    plt.title('Power Spectra Across Syllables')
    plt.axis('tight')

    ax = plt.subplot(2, 3, 3)
    plt.plot(freqs, psd_cv, 'k-', linewidth=3.0)
    plt.axis('tight')

    ax = plt.subplot(2, 3, 4)
    for dfunc in flda.discriminant_functions:
        plt.plot(freqs, dfunc, '-', linewidth=2.0, alpha=0.7)
    plt.axis('tight')
    plt.title('Discriminant Functions')

    ax = plt.subplot(2, 3, 5)
    plt.imshow(flda.scatter_between, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    plt.colorbar()
    plt.axis('tight')
    plt.title('Between Class Scatter')

    ax = plt.subplot(2, 3, 6)
    plt.imshow(flda.scatter_within, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
    plt.colorbar()
    plt.axis('tight')
    plt.title('Within Class Scatter')

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


if __name__ == '__main__':

    set_font()

    pfile = '/auto/tdrive/mschachter/data/aggregate/pairwise_cf.h5'
    agg = AggregatePairwiseCF.load(pfile)
    draw_figures(agg)

