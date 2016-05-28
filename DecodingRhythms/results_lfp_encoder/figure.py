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

    df = pd.DataFrame(pdata)
    df.to_csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder_perfs_%s.csv' % ein, index=False)

    return df


def get_encoder_weight_data_for_psd(agg):

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))
    cdata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'cell_data.csv'))

    # precompute distance from each electrode to each other electrode
    print 'Precomputing electrode distances'
    e2e_dists = dict()
    for (bird,block,hemi),gdf in edata.groupby(['bird', 'block', 'hemisphere']):

        mult = 1.
        if bird == 'GreBlu9508M':
            mult = 4.

        num_electrodes = len(gdf.electrode.unique())
        assert num_electrodes == 16
        e2e = dict()
        for e1 in gdf.electrode.unique():
            i1 = (gdf.electrode == e1)
            assert i1.sum() == 1
            dl2a1 = gdf.dist_l2a[i1].values[0] * mult
            dmid1 = gdf.dist_midline[i1].values[0]

            for e2 in gdf.electrode.unique():
                i2 = (gdf.electrode == e2)
                assert i2.sum() == 1
                dl2a2 = gdf.dist_l2a[i2].values[0] * mult
                dmid2 = gdf.dist_midline[i2].values[0]

                e2e[(e1, e2)] = np.sqrt((dl2a1 - dl2a2)**2 + (dmid1 - dmid2)**2)
        e2e_dists[(bird,block,hemi)] = e2e

    # put cell data into an efficient lookup table
    print 'Creating cell lookup table'
    cell_data = dict()
    i = cdata.cell1 == cdata.cell2
    g = cdata[i].groupby(['bird', 'block', 'segment', 'hemi', 'cell1'])
    for (bird,block,segment,hemi,ci),gdf in g:

        assert len(gdf) == 1

        # get the electrode and cell indices corresponding to this site
        wkey = '%s_%s_%s_%s_%s_%s' % (bird, block, segment, hemi, 'both', 'psd')
        index2cell = agg.index2cell[wkey]
        index2electrode = agg.index2electrode[wkey]
        cell_index2electrode = agg.cell_index2electrode[wkey]

        # get cell data
        rate = gdf.rate.values[0]
        rate_std = gdf.rate.values[0]
        cell_electrode = cell_index2electrode[ci]

        # get the distance from this cell to every other electrode
        e2e = e2e_dists[(bird,block,hemi)]
        edist = dict()
        for e in index2electrode:
            edist[e] = e2e[(cell_electrode, e)]

        cell_data[(bird,block,segment,hemi,ci)] = (rate, rate_std, edist)

    print 'Creating dataset....'
    # create the dataset
    wdata = {'bird': list(), 'block': list(), 'segment': list(), 'hemi': list(),
             'electrode': list(), 'region': list(), 'f': list(), 'w': list(), 'r2': list(),
             'dist_l2a': list(), 'dist_midline': list(), 'wtype': list(),
             'cell_index':list(),
             'rate_mean': list(), 'rate_std': list(),
             'sync_mean': list(), 'sync_std': list(),
             'dist_from_electrode': list(),
             'same_electrode':list()
             }

    i = (agg.df.encoder_input == 'both') & (agg.df.encoder_output == 'psd')
    for wkey in agg.df.wkey[i].values:
        bird, block, segment, hemi, ein2, eout2 = wkey.split('_')

        eperfs = agg.encoder_perfs[wkey]
        eweights = agg.encoder_weights[wkey]
        
        index2electrode = agg.index2electrode[wkey]
        index2cell = agg.index2cell[wkey]
        cell_index2electrode = agg.cell_index2electrode[wkey]
        ncells = len(index2cell)

        for k, e in enumerate(index2electrode):

            ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert ei.sum() == 1
            reg = clean_region(edata.region[ei].values[0])
            dist_l2a = edata.dist_l2a[ei].values[0]
            dist_midline = edata.dist_midline[ei].values[0]

            if bird == 'GreBlu9508M':
                dist_l2a *= 4

            for j, f in enumerate(agg.freqs):
                r2 = eperfs[k, j]
                W = eweights[k, j, :, :]
                assert W.shape == (ncells+1, ncells)

                # get the spike rate weights
                for n,ci in enumerate(index2cell):

                    rate,rate_std,edist = cell_data[(bird,block,segment,hemi,ci)]
                    cell_electrode = cell_index2electrode[ci]

                    wdata['bird'].append(bird)
                    wdata['block'].append(block)
                    wdata['segment'].append(segment)
                    wdata['hemi'].append(hemi)
                    wdata['electrode'].append(e)
                    wdata['region'].append(reg)
                    wdata['f'].append(int(f))
                    wdata['w'].append(W[0, n])
                    wdata['r2'].append(r2)
                    wdata['dist_l2a'].append(dist_l2a)
                    wdata['dist_midline'].append(dist_midline)
                    wdata['wtype'].append('rate')
                    wdata['rate_mean'].append(rate)
                    wdata['rate_std'].append(rate_std)
                    wdata['sync_mean'].append(-1)
                    wdata['sync_std'].append(-1)
                    wdata['dist_from_electrode'].append(edist[e])
                    wdata['cell_index'].append(ci)
                    wdata['same_electrode'].append(int(e == cell_electrode))

    wdf = pd.DataFrame(wdata)
    wdf.to_csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder_weights.csv', index=False)

    return wdf


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

    # draw_perf_by_freq(agg)
    wdf = get_encoder_weight_data_for_psd(agg)


if __name__ == '__main__':

    data_dir = '/auto/tdrive/mschachter/data'
    agg_dir = os.path.join(data_dir, 'aggregate')

    agg = LFPEncoderAggregator.load(os.path.join(agg_dir, 'lfp_encoder.h5'))
    draw_figures(agg)
