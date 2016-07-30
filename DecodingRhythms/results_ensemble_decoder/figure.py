import os
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import set_font, get_this_dir
from zeebeez.aggregators.ensemble_decoder import EnsembleDecoderAggregator
from zeebeez.utils import ACOUSTIC_PROP_NAMES


def draw_figures(agg, data_dir='/auto/tdrive/mschachter/data'):

    decomps = ['spike_rate', 'full_psds']

    aprops = ['maxAmp', 'meanspect', 'sal', 'skewspect', 'skewtime']
    curves_by_prop = dict()
    for aprop in aprops:
        for decomp in decomps:
            curves_by_prop[(decomp, aprop)] = list()

    g = agg.df.groupby(['bird', 'block', 'segment', 'aprop', 'decomp'])
    for (bird,block,segment,aprop,decomp),gdf in g:

        if aprop not in aprops:
            continue

        i = ~np.isnan(gdf.r2_mean) & ~np.isinf(gdf.r2_mean) & (gdf.r2_mean > 0)
        lst = zip(gdf.num_units[i].values, gdf.r2_mean[i].values, gdf.r2_std[i].values)
        lst.sort(key=operator.itemgetter(0))

        nu = np.array([x[0] for x in lst])
        r2 = np.array([x[1] for x in lst])
        r2_std = np.array([x[2] for x in lst])

        curves_by_prop[(decomp, aprop)].append((nu, r2, r2_std))

    for decomp in decomps:
        for aprop in aprops:
            pcurves = curves_by_prop[(decomp, aprop)]

            num_units_90 = list()
            for nu,r2,r2_std in pcurves:
                r2_q90 = 0.90*max(r2)
                nu_90 = min(nu[r2 > r2_q90])
                num_units_90.append(nu_90)

            print '%s, %s: N=%d, nu90 = %0.0f +/- %0.0f' % (decomp, aprop, len(num_units_90), np.mean(num_units_90), np.std(num_units_90, ddof=1) / np.sqrt(len(num_units_90)))

    return

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.40, wspace=0.40, left=0.05, right=0.95)

    ncols = len(aprops)
    nrows = 3
    gs = plt.GridSpec(nrows, ncols)

    for j,decomp in enumerate(decomps):
        for k,aprop in enumerate(aprops):
            ax = plt.subplot(gs[j, k])
            for nu,r2,r2_std in curves_by_prop[(decomp,aprop)]:
                plt.plot(nu, r2, 'k-', linewidth=3.0, alpha=0.7)
            plt.title(ACOUSTIC_PROP_NAMES[aprop])
            plt.axis('tight')

            if decomp == 'full_psds':
                plt.xlabel('# of Electrodes')
                plt.ylabel('R2 (LFP PSD)')
            else:
                plt.xlabel('# of Neurons')
                plt.ylabel('R2 (Spike Rate)')
                plt.xlim(1, 60)

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


if __name__ == '__main__':

    set_font()
    agg = EnsembleDecoderAggregator.load('/auto/tdrive/mschachter/data/aggregate/ensemble_decoder.h5')
    draw_figures(agg)
