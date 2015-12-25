import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder


def get_dfs(agg, data_dir='/auto/tdrive/mschachter/data'):

    # read electrode data
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    multi_electrode_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'band':list()}
    anames = agg.acoustic_props + ['category']
    for aprop in anames:
        for t in ['lfp', 'spike']:
            multi_electrode_data['perf_%s_%s' % (aprop, t)] = list()
            multi_electrode_data['lkrat_%s_%s' % (aprop, t)] = list()

    single_electrode_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(),
                             'region':list(), 'cell_index':list()}

    nbands = 12
    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])

    for (bird,block,segment,hemi),gdf in g:

        wtup = (bird,block,segment,hemi)
        index2electrode = agg.index2electrode[wtup]
        cell_index2electrode = agg.cell_index2electrode[wtup]

        band0_perfs = None
        for b in range(nbands+1):

            exfreq = b > 0

            perfs = dict()
            anames = agg.acoustic_props + ['category']
            for aprop in anames:
                for t,decomp in [('lfp', 'locked'), ('spike', 'spike_psd')]:
                    # get multielectrode LFP decoder performance
                    i = (gdf.e1 == -1) & (gdf.e2 == -1) & (gdf.cell_index == -1) & (gdf.band == b) & (gdf.exfreq == exfreq) & \
                        (gdf.exel == False) & (gdf.aprop == aprop) & (gdf.decomp == decomp)
                    assert i.sum() == 1, "Zero or more than 1 result for (%s, %s, %s, %s), decomp=locked, band=%d: i.sum()=%d" % (bird, block, segment, hemi, b, i.sum())
                    if aprop == 'category':
                        perfs['perf_%s_%s' % (aprop, t)] = gdf.pcc[i].values[0]
                    else:
                        perfs['perf_%s_%s' % (aprop, t)] = gdf.r2[i].values[0]
                    perfs['lk_%s_%s' % (aprop, t)] = gdf.likelihood[i].values[0]

            multi_electrode_data['bird'].append(bird)
            multi_electrode_data['block'].append(block)
            multi_electrode_data['segment'].append(segment)
            multi_electrode_data['hemi'].append(hemi)
            multi_electrode_data['band'].append(b)
            for k,v in perfs.items():
                if k.startswith('perf'):
                    multi_electrode_data[k].append(v)

            if b == 0:
                band0_perfs = perfs
                for aprop in anames:
                    for t in ['lfp', 'spike']:
                        multi_electrode_data['lkrat_%s_%s' % (aprop,t)].append(0)
            else:
                # compute the likelihood ratio for each acoustic property on this band
                for aprop in anames:
                    for t in ['lfp', 'spike']:
                        full_likelihood = band0_perfs['lk_%s_%s' % (aprop, t)]
                        leave_one_out_likelihood = perfs['lk_%s_%s' % (aprop, t)]
                        lkrat = 2*(leave_one_out_likelihood - full_likelihood)
                        multi_electrode_data['lkrat_%s_%s' % (aprop, t)].append(lkrat)

    df_me = pd.DataFrame(multi_electrode_data)
    df_me.to_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'), index=False)

    return df_me


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)
    df_me = get_dfs(agg)


if __name__ == '__main__':
    draw_figures()

