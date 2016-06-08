import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasp.plots import grouped_boxplot, plot_mean_from_scatter, custom_legend

from zeebeez.aggregators.cell2lfp_encoder import Cell2LFPEncoderAggregator
from DecodingRhythms.utils import clean_region, COLOR_RED_SPIKE_RATE, COLOR_CRIMSON_SPIKE_SYNC, set_font, get_this_dir, \
    get_e2e_dists


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


def get_encoder_weight_data_for_psd(agg, include_sync=True, write_to_file=True):

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))
    cdata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'cell_data.csv'))

    e2e_dists = get_e2e_dists()

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
             'dist_cell2cell':list(),
             'same_electrode':list(), 'cells_same_electrode':list(),
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
                    wdata['dist_cell2cell'].append(-1)
                    wdata['cell_index'].append(ci)
                    wdata['same_electrode'].append(int(e == cell_electrode))
                    wdata['cells_same_electrode'].append(0)

                if not include_sync:
                    continue

                # get the synchrony weights
                for n1, ci1 in enumerate(index2cell):
                    rate1, rate_std1, edist1 = cell_data[(bird, block, segment, hemi, ci1)]

                    for n2 in range(n1):
                        ci2 = index2cell[n2]
                        rate2, rate_std2, edist2 = cell_data[(bird, block, segment, hemi, ci2)]

                        e1 = cell_index2electrode[ci1]
                        e2 = cell_index2electrode[ci2]

                        cells_same_electrode = int(e1 == e2)
                        same_electrode = int(e1 == e2 and e1 == e)
                        dist_cell2cell = edist1[e2]
                        avg_dist_from_electrode = (edist1[e] + edist2[e]) / 2.

                        wdata['bird'].append(bird)
                        wdata['block'].append(block)
                        wdata['segment'].append(segment)
                        wdata['hemi'].append(hemi)
                        wdata['electrode'].append(e)
                        wdata['region'].append(reg)
                        wdata['f'].append(int(f))
                        wdata['w'].append(W[n1+1, n2])
                        wdata['r2'].append(r2)
                        wdata['dist_l2a'].append(dist_l2a)
                        wdata['dist_midline'].append(dist_midline)
                        wdata['wtype'].append('sync')
                        wdata['rate_mean'].append(-1)
                        wdata['rate_std'].append(-1)
                        wdata['sync_mean'].append(-1)
                        wdata['sync_std'].append(-1)
                        wdata['dist_from_electrode'].append(avg_dist_from_electrode)
                        wdata['dist_cell2cell'].append(dist_cell2cell)
                        wdata['cell_index'].append(-1)
                        wdata['same_electrode'].append(same_electrode)
                        wdata['cells_same_electrode'].append(cells_same_electrode)

    wdf = pd.DataFrame(wdata)
    if write_to_file:
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
            flat_data[ein].append([r2.mean(), r2.std() / np.sqrt(i.sum())])

    """
    grouped_boxplot(bp_data, group_names=[int(f) for f in agg.freqs], subgroup_names=['Rate', 'Rate+Sync'],
                    subgroup_colors=[COLOR_RED_SPIKE_RATE, COLOR_CRIMSON_SPIKE_SYNC],
                    box_width=0.6, box_spacing=1.0)
    """

    clrs = {'rate':COLOR_RED_SPIKE_RATE, 'both':COLOR_CRIMSON_SPIKE_SYNC}
    for k,ein in enumerate(etypes):
        perf = np.array(flat_data[ein])

        plt.errorbar(agg.freqs, perf[:, 0], yerr=perf[:, 1], c=clrs[ein], linewidth=8.0, elinewidth=5.0,
                     ecolor='k', alpha=0.6, capthick=0.)

    plt.legend(['Rate', 'Rate+Sync'], loc='upper right')
    plt.xlabel('LFP Frequency (Hz)')
    plt.ylabel('Mean Spike->LFP Encoder Performance (R2)')
    plt.axis('tight')
    plt.ylim(0, 1)


def draw_rate_weight_by_dist(agg):

    wdf = get_encoder_weight_data_for_psd(agg, include_sync=False, write_to_file=False)

    # plot the average encoder weight as a function of distance from predicted electrode
    clrs = {49:'k', 165:'g'}
    for f in [49, 165]:
        i = ~np.isnan(wdf.dist_from_electrode.values) & (wdf.r2 > 0.20) & (wdf.dist_from_electrode > 0) & (wdf.f == f)

        x = wdf.dist_from_electrode[i].values
        y = (wdf.w[i].values)**2

        plot_mean_from_scatter(x, y, bins=5, num_smooth_points=200, alpha=0.7, color=clrs[f], ecolor='#b5b5b5')

    plt.xlabel('Distance From Predicted Electrode (um)')
    plt.ylabel('Spike Rate Effect Size')
    plt.axis('tight')
    leg = custom_legend(colors=[clrs[49], clrs[165]], labels=['49Hz', '165Hz'])
    plt.legend(handles=leg, loc='lower left')

def draw_rate_weight_by_same(agg):
    wdf = get_encoder_weight_data_for_psd(agg, include_sync=False, write_to_file=False)

    # plot the average encoder weight as a function of distance from predicted electrode
    clrs = {49: 'k', 165: 'g'}
    vals = dict()
    for f in [49, 165]:
        i = ~np.isnan(wdf.dist_from_electrode.values) & (wdf.r2 > 0.20) & (wdf.f == f)
        df = wdf[i]

        diff_electrode = df.dist_from_electrode > 0
        same_electrode = ~diff_electrode

        diff_effect = df.w[diff_electrode].values**2
        same_effect = df.w[same_electrode].values ** 2

        vals[f] = [same_effect.mean(), diff_effect.mean(),
                   same_effect.std(ddof=1) / np.sqrt(same_electrode.sum()),
                   diff_effect.std(ddof=1) / np.sqrt(diff_electrode.sum()),]

    figsize = (5, 3)
    plt.figure(figsize=figsize)

    plt.bar([0, 0.5], vals[49][:2], yerr=vals[49][2:], width=0.45, color=clrs[49], alpha=0.7, ecolor='k')
    plt.bar([1.5, 2.0], vals[165][:2], yerr=vals[165][2:], width=0.45, color=clrs[165], alpha=0.7, ecolor='k')
    plt.xticks([])


def draw_figures(agg, data_dir='/auto/tdrive/mschachter/data'):

    figsize = (23, 8)
    fig = plt.figure(figsize=figsize)

    ax = plt.subplot(1, 2, 1)
    draw_perf_by_freq(agg)

    ax = plt.subplot(1, 2, 2)
    draw_rate_weight_by_dist(agg)

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    draw_rate_weight_by_same(agg)
    fname = os.path.join(get_this_dir(), 'figure_inset.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()

    # wdf = get_encoder_weight_data_for_psd(agg)


if __name__ == '__main__':

    set_font()

    data_dir = '/auto/tdrive/mschachter/data'
    agg_dir = os.path.join(data_dir, 'aggregate')

    agg = Cell2LFPEncoderAggregator.load(os.path.join(agg_dir, 'cell2lfp_encoder.h5'))
    draw_figures(agg)
