import os
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import COLOR_RED_SPIKE_RATE, set_font, get_this_dir
from lasp.colormaps import magma
from zeebeez.aggregators.tuning_curve import TuningCurveAggregator
from zeebeez.utils import ACOUSTIC_PROP_NAMES, USED_ACOUSTIC_PROPS


def draw_r2(agg, ax=None):

    freqs = list(sorted(agg.df.freq.unique().astype('int')))
    aprops = USED_ACOUSTIC_PROPS

    mean_r2 = np.zeros([len(aprops), len(freqs)])

    for k, aprop in enumerate(aprops):

        for j,f in enumerate(freqs):

            decomp = 'full_psds'
            if f == -1:
                decomp = 'spike_rate'

            i = (agg.df.decomp == decomp) & (agg.df.freq == f) & (agg.df.aprop == aprop) & (agg.df.r2 > 0)
            if i.sum() == 0:
                print 'no datapoints for aprop=%s, decomp=%s, freq=%d' % (aprop, decomp, f)
            else:
                r2 = agg.df.r2[i]
                mean_r2[k, j] = r2.mean()

    if ax is None:
        plt.figure()
        ax = plt.gca()

    x_labels = ['Spike\nRate']
    for f in freqs:
        if f == -1:
            continue
        x_labels.append('%d' % f)

    mean_r2,new_order = reorder_by_row_sum(mean_r2)

    aprop_labels = [ACOUSTIC_PROP_NAMES[aprop] for aprop in np.array(USED_ACOUSTIC_PROPS)[new_order]]

    r2max = mean_r2.max()
    plt.imshow(mean_r2, interpolation='nearest', aspect='auto', cmap=magma, vmin=0, vmax=r2max)
    cbar = plt.colorbar(label='Mean Tuning Curve R2')
    new_ytks = ['%0.2f' % float(_yt.get_text()) for _yt in cbar.ax.get_yticklabels()]
    cbar.ax.set_yticklabels(new_ytks)
    plt.yticks(np.arange(len(USED_ACOUSTIC_PROPS)), aprop_labels)
    plt.xticks(np.arange(len(x_labels)), x_labels)
    plt.xlabel('Frequency (Hz)')


def reorder_by_row_sum(W):
    n = W.shape[0]
    rsum = W.sum(axis=1)
    new_order = list(sorted(zip(range(n), rsum), key=operator.itemgetter(1), reverse=True))
    Wr = np.array([W[k[0], :] for k in new_order])
    print 'Wr.shape=',Wr.shape
    return Wr,[x[0] for x in new_order]


def draw_curves(agg):

    aprops = ['maxAmp', 'meanspect', 'entropytime', 'sal']
    freqs = [0, 33, 182]
    decomps = (('spike_rate', -1), ('full_psds', freqs[0]), ('full_psds', freqs[1]), ('full_psds', freqs[2]))

    # get top tuning cuves for each acoustic prop
    plot_perfs = False
    if plot_perfs:
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        nrows = 5
        ncols = 3

    perf_thresh = 0.05
    top_tuning_cuves = dict()
    for k,aprop in enumerate(aprops):
        for j,(decomp,f) in enumerate(decomps):

            i = (agg.df.decomp == decomp) & (agg.df.freq == f) & (agg.df.aprop == aprop)
            assert i.sum() > 0, 'aprop=%s, decomp=%s, freq=%d' % (aprop, decomp, f)
            perf = agg.df.r2[i].values
            assert np.sum(np.isnan(perf)) == 0
            assert np.sum(np.isinf(perf)) == 0

            pi = perf > perf_thresh
            xindex = agg.df.xindex[i][pi]
            if len(xindex) == 0:
                print 'No good curves for aprop=%s, decomp=%s, freq=%d' % (aprop, decomp, f)
                continue

            lst = zip(xindex, perf[pi])
            lst.sort(key=operator.itemgetter(1), reverse=True)
            xindex = [x[0] for x in lst]
            alpha = np.array([x[1] for x in lst])
            alpha -= alpha.min()
            alpha /= alpha.max()

            cx = agg.curve_x[xindex, :]
            tc = agg.tuning_curves[xindex, :]

            good_indices = np.ones([cx.shape[0]], dtype='bool')
            if decomp == 'spike_rate':
                for k in range(cx.shape[0]):
                    good_indices[k] = ~np.any(tc[k, :] > 50)

            top_tuning_cuves[(aprop, decomp, f)] = (cx[good_indices, :], tc[good_indices, :], alpha[good_indices])

            if plot_perfs:
                ax = plt.subplot(nrows, ncols, k*len(decomps) + j + 1)
                plt.hist(perf[perf > 0.6], bins=20)
                plt.title('%s, %s, %d' % (aprop, decomp, f))
                plt.axis('tight')

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.99, left=0.05, hspace=0.25, wspace=0.25)

    gs = plt.GridSpec(100, 100)

    offset = 55
    wpad = 5
    hpad = 8
    height = int(100 / len(aprops)) - hpad
    width = int((100 - offset) / 3.) - wpad

    topn = 30

    xticks = {'meanspect':([2, 4], ['2', '4']),
              'maxAmp':([0.2, 0.4], ['0.2', '0.4']),
              'sal':([0.5, 0.7], ['0.5', '0.7']),
              'entropytime':([0.90, 0.94, 0.98], ['0.90', '0.94', '0.98'])
             }

    clrs = {('spike_rate', -1):COLOR_RED_SPIKE_RATE, ('full_psds', 0):'k', ('full_psds', 33):'k', ('full_psds', 182):'k'}

    for k, aprop in enumerate(aprops):
        for j, (decomp, f) in enumerate(decomps):

            x1 = j*(width+wpad)
            x2 = x1 + width
            y1 = k*(height+hpad)
            y2 = y1 + height
            # print 'k=%d, j=%d, x1=%d, x2=%d, y1=%d, y2=%d' % (k, j, x1, x2, y1, y2)

            ax = plt.subplot(gs[y1:y2, x1:x2])
            # plot the top n tuning curves
            if (aprop, decomp, f) not in top_tuning_cuves:
                continue

            cx,tc,alpha = top_tuning_cuves[(aprop, decomp, f)]
            if aprop == 'meanspect':
                cx *= 1e-3
            n = min(cx.shape[0], 90)
            plt.axhline(0, c='k')
            for x,y,a in zip(cx[:n, :], tc[:n, :], alpha[:n]):
                c = clrs[(decomp, f)]
                plt.plot(x, y, '-', c=c, linewidth=1.0, alpha=a)

                xlbl = ACOUSTIC_PROP_NAMES[aprop]
                if aprop == 'meanspect':
                    xlbl += ' (kHz)'
                elif aprop == 'entropytime':
                    xlbl += '(bits)'

                plt.xlabel(xlbl)

                if decomp.endswith('rate'):
                    ylbl = 'Spike Rate\n(z-scored)'
                    if k == 0:
                        plt.title('Spike Rate', fontweight='bold')
                elif decomp.endswith('psds'):
                    if k == 0:
                        plt.title('LFP Power (%d Hz)' % f, fontweight='bold')
                    ylbl = 'LFP Power'

                plt.ylabel(ylbl)
                if aprop in xticks:
                    plt.xticks(xticks[aprop][0], xticks[aprop][1])

            plt.axis('tight')
            if aprop == 'entropytime':
                plt.xlim(0.89, 0.98)
            else:
                plt.xlim([x.min() * 1.20, x.max() * 0.9])
            if decomp.endswith('rate'):
                plt.ylim(-1.5, 1.5)
            elif decomp.endswith('psds'):
                plt.ylim(-1., 1.)

    ax = plt.subplot(gs[:45, 62:])
    draw_r2(agg, ax)

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'tuning_curve.h5')
    agg = TuningCurveAggregator.load(agg_file)

    draw_curves(agg)
    # draw_r2(agg)
    plt.show()


if __name__ == '__main__':

    set_font()
    draw_figures()


