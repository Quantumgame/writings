import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasp.plots import grouped_boxplot

from zeebeez.aggregators.lfp_encoder import LFPEncoderAggregator
from utils import clean_region, COLOR_RED_SPIKE_RATE, COLOR_CRIMSON_SPIKE_SYNC


def get_encoder_perf_data_for_psd(agg, ein='rate'):

    i = (agg.df.encoder_input == ein) & (agg.df.encoder_output == 'psd')

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))


    pdata = {'bird': list(), 'block': list(), 'segment': list(), 'hemi': list(),
             'electrode': list(), 'region': list(), 'f': list(), 'r2': list(),
             'dist_l2a': list(), 'dist_midline': list()}

    for wkey in agg.df.wkey[i].values:
        bird, block, segment, hemi, ein2, eout2 = wkey.split('_')

        eperfs = agg.encoder_perfs[wkey]
        index2electrode = agg.index2electrode[wkey]

        for k, e in enumerate(index2electrode):

            ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert ei.sum() == 1
            reg = clean_region(edata.region[ei].values[0])
            dist_l2a = edata.dist_l2a[ei].values[0]
            dist_midline = edata.dist_midline[ei].values[0]

            if bird == 'GreBlu9508M':
                dist_l2a *= 4

            for j, f in enumerate(agg.freqs):
                pdata['bird'].append(bird)
                pdata['block'].append(block)
                pdata['segment'].append(segment)
                pdata['hemi'].append(hemi)
                pdata['electrode'].append(e)

                pdata['region'].append(reg)
                pdata['dist_l2a'].append(dist_l2a)
                pdata['dist_midline'].append(dist_midline)

                pdata['f'].append(int(f))
                pdata['r2'].append(eperfs[k, j])

    return pd.DataFrame(pdata)


def draw_perf_by_freq(agg, data_dir='/auto/tdrive/mschachter/data'):

    etypes = ['rate', 'both']
    df = dict()
    for ein in etypes:
        df[ein] = get_encoder_perf_data_for_psd(agg, ein)

    flat_data = dict()
    for ein in etypes:
        flat_data[ein] = list()
    bp_data = dict()
    for f in [int(x) for x in agg.freqs]:
        bp_data[f] = list()
        for ein in etypes:
            i = df[ein].f == f
            r2 = df[ein].r2[i].values
            bp_data[f].append(r2)
            flat_data[ein].append([r2.mean(), r2.std()])

    """
    grouped_boxplot(bp_data, group_names=[int(f) for f in agg.freqs], subgroup_names=['Rate', 'Rate+Sync'],
                    subgroup_colors=[COLOR_RED_SPIKE_RATE, COLOR_CRIMSON_SPIKE_SYNC],
                    box_width=0.6, box_spacing=1.0)
    """

    clrs = {'rate':COLOR_RED_SPIKE_RATE, 'both':COLOR_CRIMSON_SPIKE_SYNC}
    plt.figure()
    for ein in etypes:
        perf = np.array(flat_data[ein])
        plt.plot(agg.freqs, perf[:, 0], '-', c=clrs[ein], linewidth=5.0, alpha=0.7)

    plt.legend(['Rate', 'Rate+Sync'])
    plt.xlabel('LFP Frequency (Hz)')
    plt.ylabel('Encoder Performance')
    plt.axis('tight')
    plt.show()


def draw_figures(agg, data_dir='/auto/tdrive/mschachter/data'):

    draw_perf_by_freq(agg)


if __name__ == '__main__':

    data_dir = '/auto/tdrive/mschachter/data'
    agg_dir = os.path.join(data_dir, 'aggregate')

    agg = LFPEncoderAggregator.load(os.path.join(agg_dir, 'lfp_encoder.h5'))
    draw_figures(agg)
