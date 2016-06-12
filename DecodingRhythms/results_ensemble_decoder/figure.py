import os
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import set_font, get_this_dir
from zeebeez.aggregators.ensemble_decoder import EnsembleDecoderAggregator
from zeebeez.utils import ACOUSTIC_PROP_NAMES


def draw_figures(agg, data_dir='/auto/tdrive/mschachter/data'):

    i = agg.df.decomp == 'full_psds'

    aprops = ['entropytime', 'sal', 'meanspect', 'maxAmp', 'stdtime']
    curves_by_prop = dict()
    for aprop in aprops:
        curves_by_prop[aprop] = list()

    df = agg.df[i]
    g = df.groupby(['bird', 'block', 'segment', 'aprop'])
    for (bird,block,segment,aprop),gdf in g:

        lst = zip(gdf.num_units.values, gdf.r2_mean.values, gdf.r2_std.values)
        lst.sort(key=operator.itemgetter(0))

        nu = [x[0] for x in lst]
        r2 = [x[1] for x in lst]
        r2_std = [x[2] for x in lst]

        curves_by_prop[aprop].append((nu, r2, r2_std))

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.40, wspace=0.40, left=0.05, right=0.95)

    ncols = len(aprops)
    nrows = 3
    gs = plt.GridSpec(nrows, ncols)

    for k,aprop in enumerate(aprops):
        ax = plt.subplot(gs[0, k])
        for nu,r2,r2_std in curves_by_prop[aprop]:
            plt.plot(nu, r2, 'k-', linewidth=3.0, alpha=0.7)
        plt.title(ACOUSTIC_PROP_NAMES[aprop])
        plt.xlabel('# of Electrodes')
        plt.ylabel('Decoder R2')
        plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


if __name__ == '__main__':

    set_font()
    agg = EnsembleDecoderAggregator.load('/auto/tdrive/mschachter/data/aggregate/ensemble_decoder.h5')
    draw_figures(agg)
