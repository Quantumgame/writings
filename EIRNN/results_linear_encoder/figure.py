import os
import operator

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from DecodingRhythms.utils import set_font, get_this_dir
from lasp.plots import boxplot_with_colors
from zeebeez.aggregators.lfp_encoder import AggregateLFPEncoder


def draw_perfs(agg):

    # filter out some data points
    i = (agg.df.cc > 0.05) & (agg.df.region != '?') & (agg.df.region != 'HP') \
        & ~np.isnan(agg.df.dist_l2a) & ~np.isnan(agg.df.dist_midline) \
        & (agg.df.dist_l2a > -1) & (agg.df.dist_l2a < 1)
    df = agg.df[i]

    print len(df)
    print df.region.unique()

    # make a scatter plot of performance by anatomical location
    fig = plt.figure(figsize=(23, 10), facecolor='w')
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.25)

    gs = plt.GridSpec(2, 100)

    ax = plt.subplot(gs[:, :55])
    plt.scatter(df.dist_midline, df.dist_l2a, c=df.cc, s=64, cmap=plt.cm.afmhot_r, alpha=0.7)
    plt.ylim(-1, 1)
    plt.xlim(0, 2.5)
    for x, y, r in zip(df.dist_midline, df.dist_l2a, df.region):
        plt.text(x, y - 0.05, r, fontsize=9)
    cbar = plt.colorbar(label='Linear Encoder Performance (cc)')
    plt.xlabel('Dist to Midline (mm)')
    plt.ylabel('Dist to L2A (mm)')

    region_stats = list()
    regions_to_use = ['L1', 'L2', 'L3', 'CM', 'NCM']
    for reg in regions_to_use:
        i = df.region == reg
        region_stats.append({'cc': df[i].cc, 'cc_mean': df[i].cc.mean(), 'region': reg})
    region_stats.sort(key=operator.itemgetter('cc_mean'), reverse=True)
    regions = [x['region'] for x in region_stats]
    region_ccs = {x['region']: x['cc'] for x in region_stats}

    ax = plt.subplot(gs[:45, 65:])
    boxplot_with_colors(region_ccs, group_names=regions, ax=ax, group_colors=['k'] * len(regions), box_alpha=0.7)
    plt.xlabel('Region')
    plt.ylabel('Linear Encoder Performance (cc)')

    fname = os.path.join(get_this_dir(), 'encoder_perf.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')


def draw_filters(agg):

    # filter out some data points
    i = (agg.df.cc > 0.20) & (agg.df.region != '?') & (agg.df.region != 'HP')
    df = agg.df[i]

    regions_to_use = ['L1', 'L2', 'L3', 'CM', 'NCM']
    region_filters = list()
    lags_ms = (agg.lags/agg.sample_rate)*1e3
    for reg in regions_to_use:
        i = df.region == reg
        xi = df.xindex[i].values
        filts = agg.filters[xi, :]
        # quantify peak time
        filt_peaks = compute_filt_peaks(filts, lags_ms)
        region_filters.append({'filters':filts, 'region':reg, 'peak_mean':filt_peaks.mean()})

    region_filters.sort(key=operator.itemgetter('peak_mean'))

    topn = 20
    lag_i = (lags_ms < 100.)
    fig = plt.figure(figsize=(10, 12), facecolor='w')
    fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.95, bottom=0.05)
    nrows = len(region_filters)
    for k,rdict in enumerate(region_filters):
        ax = plt.subplot(nrows, 1, k+1)
        plt.axhline(0, c='k')
        for f in rdict['filters'][:topn]:
            plt.plot(lags_ms[lag_i], f[lag_i]*1e3, 'k-', alpha=0.7, linewidth=2.0)
        plt.axis('tight')
        plt.ylim(-4, 4)
        plt.title(rdict['region'])
        if k == len(region_filters)-1:
            plt.xlabel('Filter Lag (ms)')

    fname = os.path.join(get_this_dir(), 'encoder_filters.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()

def compute_filt_peaks(filts, lags_ms):

    peaks = np.zeros(filts.shape[0])
    for k,f in enumerate(filts):
        npeak = np.argmin(f)
        peaks[k] = lags_ms[npeak]
    return peaks


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_encoder.h5')
    agg = AggregateLFPEncoder.load(agg_file)

    agg.df.to_csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder.csv', header=True, index=False)

    # draw_perfs(agg)

    draw_filters(agg)

    plt.show()


if __name__ == '__main__':

    set_font()
    draw_figures()


