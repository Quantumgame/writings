import os
import h5py

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from lasp.plots import multi_plot

from zeebeez.aggregators.acoustic_encoder_decoder import AcousticEncoderDecoderAggregator
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, REDUCED_ACOUSTIC_PROPS


def draw_encoder_perfs(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder_pairwise'):

    #TODO: hack
    # read the lags
    pfile = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    hf = h5py.File(pfile, 'r')
    lags_ms = hf.attrs['lags']
    hf.close()

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = AcousticEncoderDecoderAggregator.load(agg_file)

    i = agg.df.decomp == 'self+cross_locked'

    for (bird,block,segment,hemi),gdf in agg.df[i].groupby(['bird', 'block', 'segment','hemi']):

        assert len(gdf) == 1

        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT
        if hemi == 'L':
            electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT

        # get segment/decomp props
        wkey = gdf.wkey.values[0]
        iindex = gdf.iindex.values[0]

        # get encoder performances
        encoder_perfs = agg.encoder_perfs[wkey]
        index2electrode = agg.index2electrode[iindex]
        assert len(lags_ms) == encoder_perfs.shape[-1]
        nelectrodes = len(index2electrode)
        r2_max = 0.3

        # get decoder weights
        decoder_weights = agg.decoder_weights[wkey]
        dperfs = agg.decoder_perfs[wkey]

        for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
            dW = decoder_weights[:, :, :, k]
            wmin = dW.min()
            wmax = dW.max()

            # make a figure for encoder performance
            figsize = (24, 13)
            fig = plt.figure(figsize=figsize)
            fig.subplots_adjust(top=0.95, bottom=0.06, right=0.97, left=0.03, hspace=0.25, wspace=0.25)
            gs = plt.GridSpec(nelectrodes, nelectrodes)

            for i in range(nelectrodes):
                for j in range(i):

                    eperf = encoder_perfs[i, j, :]
                    w = dW[i, j, :]

                    ax = plt.subplot(gs[i, j])
                    for li,er2 in enumerate(eperf):
                        a = (er2 / r2_max)*0.85
                        plt.axvline(lags_ms[li], c='r', alpha=a, linewidth=2.0)
                    plt.axvline(0, c='k', alpha=0.9)
                    # plt.axvline(-5., c='k', alpha=0.5)
                    # plt.axvline(5., c='k', alpha=0.5)
                    plt.axhline(0, c='k', alpha=0.5)
                    plt.plot(lags_ms, w, 'k-', linewidth=3.0, alpha=0.8)
                    plt.axis('tight')
                    plt.ylim(wmin, wmax)

                    plt.xticks([])
                    plt.yticks([])

                    if j == 0:
                        ytks = [wmin, 0, wmax]
                        plt.yticks(ytks, ['%0.2f' % x for x in ytks])
                        plt.ylabel('E%d' % electrode_order[i])
                    if i == nelectrodes-1:
                        xtks = [-40., 0, 40]
                        plt.xticks(xtks, ['%d' % x for x in xtks])
                        plt.xlabel('E%d' % electrode_order[j])

            fname = '%s_%s_%s_%s_decoder_%s' % (bird, block, segment, hemi, aprop)
            plt.suptitle('%s (R2=%0.2f)' % (fname, dperfs[k]))
            fpath = os.path.join(fig_dir, '%s.svg' % fname)
            print 'Saving %s...' % fpath
            plt.savefig(fpath, facecolor='w')
            plt.close('all')

        # make a figure for encoder performance
        figsize = (24, 13)
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(top=0.95, bottom=0.06, right=0.97, left=0.03, hspace=0.25, wspace=0.25)
        gs = plt.GridSpec(nelectrodes, nelectrodes)

        for i in range(nelectrodes):
            for j in range(i):

                perf = encoder_perfs[i, j, :]
                ax = plt.subplot(gs[i, j])
                plt.axvline(0, c='k', alpha=0.5)
                plt.plot(lags_ms, perf, 'k-', linewidth=2.0, alpha=0.7)
                plt.axis('tight')

                plt.xticks([])
                plt.yticks([])

                if j == 0:
                    ytks = [r2_max/3., (2/3.)*r2_max, r2_max]
                    plt.ylim(0, r2_max)
                    plt.yticks(ytks, ['%0.2f' % x for x in ytks])
                    plt.ylabel('E%d' % electrode_order[i])
                if i == nelectrodes-1:
                    xtks = [-40., 0, 40]
                    plt.xticks(xtks, ['%d' % x for x in xtks])
                    plt.xlabel('E%d' % electrode_order[j])

        fname = '%s_%s_%s_%s_encoder' % (bird, block, segment, hemi)
        plt.suptitle(fname)
        fname = os.path.join(fig_dir, '%s.svg' % fname)
        print 'Saving %s...' % fname
        plt.savefig(fname, facecolor='w')
        plt.close('all')


def draw_lags_vs_perf(data_dir='/auto/tdrive/mschachter/data'):

    pfile = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    hf = h5py.File(pfile, 'r')
    full_lags_ms = hf.attrs['lags']
    hf.close()

    max_lag = full_lags_ms.max()

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = AcousticEncoderDecoderAggregator.load(agg_file)

    print 'keys=',agg.df.keys()

    lag_bnds = [ (-1., 1.), (-6., 6.), (-13., 13.), (-18., 18), (-23., 23), (-28., 28.), (-34., 34), (None,None)]
    perfs_by_bound = list()
    for lb,ub in lag_bnds:
        if lb is None:
            decomp = 'self+cross_locked'
        else:
            decomp = 'self+cross_locked_lim_%d_%d' % (lb, ub)

        i = agg.df.decomp == decomp
        print 'decomp=%s, i.sum()=%d' % (decomp, i.sum())
        wkeys = agg.df.wkey[i].values

        dperfs = np.array([agg.decoder_perfs[wkey] for wkey in wkeys])
        perfs_by_bound.append(dperfs)

    perfs_by_bound = np.array(perfs_by_bound)

    pbb_mean = perfs_by_bound.mean(axis=1)
    pbb_std = perfs_by_bound.std(axis=1)

    plist = list()
    for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
        plist.append({'aprop':aprop, 'mean':pbb_mean[:, k], 'std':pbb_std[:, k]})

    def _plot_pbb(pdata, ax):
        _lagw = [2*x[1] for x in lag_bnds if x[1] is not None]
        _lagw.append(max_lag*2)
        plt.plot(_lagw, pdata['mean'], 'k-', linewidth=3.0, alpha=0.7)
        plt.xlabel('Lag Width (ms)')
        plt.ylabel('R2')
        plt.axis('tight')
        plt.title(pdata['aprop'])

    multi_plot(plist, _plot_pbb, nrows=3, ncols=4)
    plt.show()


def build_graph(bird='GreBlu9508M', block='Site4', segment='Call1', hemi='L', aprop='q2'):

    # map electrodes to positions
    df = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/electrode_data+dist.csv')
    i = (df.bird == bird) & (df.block == block) & (df.hemisphere == hemi)
    g = df[i].groupby('electrode')
    electrode_props = dict()
    for e,gdf in g:
        assert len(gdf) == 1
        reg = str(gdf.region.values[0])
        if reg.startswith('CM'):
            reg = 'CM'
        elif 'NCM' in reg:
            reg = 'NCM'
        elif reg in ['L', 'L2A', 'L2B', 'L2', 'L1', 'L3']:
            reg = 'L'
        electrode_props[e] = {'dist_midline':float(gdf.dist_midline.values[0]),
                              'dist_l2a':float(gdf.dist_l2a.values[0]),
                              'region':reg
                              }

    #TODO get the lags in a less hacky way
    data_dir = '/auto/tdrive/mschachter/data'
    pfile = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    hf = h5py.File(pfile, 'r')
    lags_ms = hf.attrs['lags']
    hf.close()

    lags_absmax = 6.
    lags_i = np.abs(lags_ms) < lags_absmax
    lags_short = lags_ms[lags_i]

    # read the aggregator file
    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = AcousticEncoderDecoderAggregator.load(agg_file)

    i = (agg.df.decomp == 'self+cross_locked_lim_-%d_%d' % (int(lags_absmax), int(lags_absmax))) & (agg.df.bird == bird) & (agg.df.block == block) & \
        (agg.df.segment == segment) & (agg.df.hemi == hemi)
    print 'i.sum()=%d' % i.sum()
    assert i.sum() == 1, 'i.sum()=%d' % i.sum()

    # get decoder weights
    electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT
    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    electrode_order = [int(x) for x in electrode_order]

    # get segment/decomp props
    wkey = agg.df[i].wkey.values[0]
    iindex = agg.df[i].iindex.values[0]

    # get decoder weights
    decoder_weights = agg.decoder_weights[wkey]
    W = decoder_weights[:, :, :, REDUCED_ACOUSTIC_PROPS.index(aprop)]

    # create a directed graph
    g = nx.DiGraph()

    # add a node for each electrode
    for e in electrode_order:
        g.add_node(e, dist_midline=electrode_props[e]['dist_midline'], dist_l2a=electrode_props[e]['dist_l2a'],
                   region=electrode_props[e]['region'], electrode=e)

    # connect up the nodes
    for k,e1 in enumerate(electrode_order):
        for j in range(k):
            e2 = electrode_order[j]

            # get the coherency function and reduce it to a left and right side
            cf = W[k, j, :]
            assert len(cf) == len(lags_short)
            cleft = np.sum(cf[lags_short <= 0]) # goes from e2 to e1
            cright = np.sum(cf[lags_short >= 0]) # goes from e1 to e2

            # add edges to the graph
            g.add_edge(e2, e1, weight=float((cleft)))
            g.add_edge(e1, e2, weight=float((cright)))

    # plot the edge weights
    ewts = np.array([g.edge[e1][e2]['weight'] for e1,e2 in g.edges()])

    plt.figure()
    plt.hist(ewts, bins=20, color='c')
    plt.xlabel('Edge Weight')
    plt.show()

    # threshold out the edges with low weights
    wthresh = 0.3
    for e1,e2 in [(e1,e2) for e1,e2 in g.edges() if abs(g.edge[e1][e2]['weight']) < wthresh]:
        g.remove_edge(e1, e2)

    # write the graph to a file
    nx.write_gexf(g, '/tmp/%s_graph.gexf' % aprop)


def draw_figures():
    # draw_encoder_perfs()
    draw_lags_vs_perf()
    # build_graph()


if __name__ == '__main__':
    draw_figures()
