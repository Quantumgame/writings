import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, REDUCED_ACOUSTIC_PROPS


def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder_pairwise'):

    #TODO: hack
    # read the lags
    pfile = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    hf = h5py.File(pfile, 'r')
    lags_ms = hf.attrs['lags']
    hf.close()

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)

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
                        a = (er2 / r2_max)*0.7
                        plt.axvline(lags_ms[li], c='r', alpha=a, linewidth=2.0)
                    plt.axvline(0, c='k', alpha=0.5)
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
            fname = os.path.join(fig_dir, '%s.png' % fname)
            print 'Saving %s...' % fname
            plt.savefig(fname, facecolor='w')
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


if __name__ == '__main__':

    draw_figures()