import os
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import COLOR_RED_SPIKE_RATE, set_font, get_this_dir
from zeebeez.aggregators.tuning_curve import TuningCurveAggregator
from zeebeez.utils import ACOUSTIC_PROP_NAMES


def draw_curves(agg):

    aprops = ['maxAmp', 'sal', 'meanspect', 'entropytime']
    freqs = [49, 165]
    decomps = (('spike_rate', -1), ('full_psds', freqs[0]), ('full_psds', freqs[1]))

    # get top tuning cuves for each acoustic prop
    plot_perfs = False
    if plot_perfs:
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.35, hspace=0.35)
        nrows = 5
        ncols = 3

    perf_thresh = 0.95
    top_tuning_cuves = dict()
    for k,aprop in enumerate(aprops):
        for j,(decomp,f) in enumerate(decomps):

            i = (agg.df.decomp == decomp) & (agg.df.freq == f) & (agg.df.aprop == aprop)
            assert i.sum() > 0, 'aprop=%s, decomp=%s, freq=%d' % (aprop, decomp, f)
            perf = agg.df.mse_nonlin[i].values / agg.df.mse_mean[i].values
            perf[np.isnan(perf)] = 1.
            perf[np.isinf(perf)] = 1.

            pi = perf < perf_thresh
            xindex = agg.df.xindex[i][pi]
            if len(xindex) == 0:
                print 'No good curves for aprop=%s, decomp=%s, freq=%d' % (aprop, decomp, f)
                continue

            lst = zip(xindex, perf[pi])
            lst.sort(key=operator.itemgetter(1))
            xindex = [x[0] for x in lst]

            cx = agg.curve_x[xindex, :]
            tc = agg.tuning_curves[xindex, :]

            good_indices = np.ones([cx.shape[0]], dtype='bool')
            if decomp == 'spike_rate':
                for k in range(cx.shape[0]):
                    good_indices[k] = ~np.any(tc[k, :] > 50)

            top_tuning_cuves[(aprop, decomp, f)] = (cx[good_indices, :], tc[good_indices, :])

            if plot_perfs:
                ax = plt.subplot(nrows, ncols, k*len(decomps) + j + 1)
                plt.hist(perf[perf > 0.6], bins=20)
                plt.title('%s, %s, %d' % (aprop, decomp, f))
                plt.axis('tight')

    figsize = (23, 10)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.25, wspace=0.25)

    gs = plt.GridSpec(100, 100)

    offset = 45
    wpad = 5
    hpad = 8
    height = int(100 / len(aprops)) - hpad
    width = int((100 - offset) / 3.) - wpad

    topn = 30

    xticks = {'meanspect':([1, 3, 5], ['1', '3', '5']),
              'maxAmp':([0.1, 0.4, 0.7], ['0.1', '0.4', '0.7']),
              'sal':([0.3, 0.6, 0.8], ['0.3', '0.6', '0.8']),
             }

    clrs = {('spike_rate', -1):COLOR_RED_SPIKE_RATE, ('full_psds', 49):'k', ('full_psds', 165):'g'}

    for k, aprop in enumerate(aprops):
        for j, (decomp, f) in enumerate(decomps):

            x1 = j*(width+wpad) + offset
            x2 = x1 + width
            y1 = k*(height+hpad)
            y2 = y1 + height
            # print 'k=%d, j=%d, x1=%d, x2=%d, y1=%d, y2=%d' % (k, j, x1, x2, y1, y2)

            ax = plt.subplot(gs[y1:y2, x1:x2])
            # plot the top n tuning curves
            if (aprop, decomp, f) not in top_tuning_cuves:
                continue

            cx,tc = top_tuning_cuves[(aprop, decomp, f)]
            if aprop == 'meanspect':
                cx *= 1e-3
            n = min(topn, cx.shape[0])
            for x,y in zip(cx[:n, :], tc[:n, :]):
                c = clrs[(decomp, f)]
                plt.plot(x, y, '-', c=c, linewidth=1.0, alpha=0.7)
                plt.xlabel(ACOUSTIC_PROP_NAMES[aprop])
                if decomp.endswith('rate'):
                    ylbl = 'Spike Rate (Hz)'
                    if k == 0:
                        plt.title('Spike Rate')
                elif decomp.endswith('psds'):
                    if k == 0:
                        plt.title('LFP Power (%d Hz)' % f)
                    ylbl = 'LFP Power'
                plt.ylabel(ylbl)
                if aprop in xticks:
                    plt.xticks(xticks[aprop][0], xticks[aprop][1])
            plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'tuning_curve.h5')
    agg = TuningCurveAggregator.load(agg_file)

    draw_curves(agg)


if __name__ == '__main__':

    set_font()
    draw_figures()


