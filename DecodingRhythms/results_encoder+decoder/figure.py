import os
from copy import deepcopy

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasp.plots import custom_legend

from DecodingRhythms.utils import set_font, get_this_dir
from lasp.colormaps import magma

from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import REDUCED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, \
    ACOUSTIC_FEATURE_COLORS


def draw_encoder_perfs(agg):

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    freqs = hf.attrs['freqs']
    hf.close()

    electrodes = [ROSTRAL_CAUDAL_ELECTRODES_LEFT,
                  ROSTRAL_CAUDAL_ELECTRODES_RIGHT,
                  ROSTRAL_CAUDAL_ELECTRODES_LEFT]

    sites = [('GreBlu9508M', 'Site4', 'Call1', 'L'),
             ('WhiWhi4522M', 'Site2', 'Call2', 'R'),
             ('YelBlu6903F', 'Site3', 'Call3', 'L')
             ]

    weights = list()
    for bird,block,segment,hemi in sites:

        i = (agg.df.bird == bird) & (agg.df.block == block) & (agg.df.segment == segment) & (agg.df.hemi == hemi)

        ii = i & (agg.df.decomp == 'self_locked')
        assert ii.sum() == 1
        wkey = agg.df[ii]['wkey'].values[0]
        lfp_eperf = agg.encoder_perfs[wkey]

        lfp_decoder_weights = agg.decoder_weights[wkey]
        print 'lfp_decoder_weights.shape=',lfp_decoder_weights.shape
        lfp_sal_weights = lfp_decoder_weights[:, :, REDUCED_ACOUSTIC_PROPS.index('sal')]
        lfp_q2_weights = lfp_decoder_weights[:, :, REDUCED_ACOUSTIC_PROPS.index('q2')]

        weights.append( (lfp_eperf, lfp_sal_weights, lfp_q2_weights) )

    figsize = (24, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.25, wspace=0.25)
    nrows = 3
    ncols = 3

    for k,(lfp_eperf,lfp_sal_weights,lfp_q2_weights) in enumerate(weights):

        index2electrode = electrodes[k]

        ax = plt.subplot(nrows, ncols, k*ncols + 1)
        plt.imshow(lfp_eperf, interpolation='nearest', aspect='auto', vmin=0, cmap=magma)
        plt.yticks(range(len(index2electrode)), ['%d' % e for e in index2electrode])
        plt.xticks(range(len(freqs)), ['%d' % f for f in freqs], rotation=45)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Electrode')
        plt.colorbar(label='Encoder R2')

        ax = plt.subplot(nrows, ncols, k*ncols + 2)
        absmax = np.abs(lfp_sal_weights).max()
        plt.imshow(lfp_sal_weights, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.yticks(range(len(index2electrode)), ['%d' % e for e in index2electrode])
        plt.xticks(range(len(freqs)), ['%d' % f for f in freqs], rotation=45)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Electrode')
        plt.colorbar(label='Decoder Weight')

        ax = plt.subplot(nrows, ncols, k*ncols + 3)
        absmax = np.abs(lfp_sal_weights).max()
        plt.imshow(lfp_q2_weights, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
        plt.yticks(range(len(index2electrode)), ['%d' % e for e in index2electrode])
        plt.xticks(range(len(freqs)), ['%d' % f for f in freqs], rotation=45)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Electrode')
        plt.colorbar(label='Decoder Weight')

    fname = os.path.join(get_this_dir(), 'encoder_weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_encoder_weights(agg):

    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    freqs = hf.attrs['freqs']
    hf.close()

    aprops = REDUCED_ACOUSTIC_PROPS

    w_by_decomp = dict()

    for decomp in ['self_locked', 'self_spike_psd']:
        all_weights = list()
        i = agg.df.decomp == decomp
        g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
        for (bird,block,esgment,hemi),gdf in g:
            assert len(gdf) == 1
            wkey = gdf.wkey.values[0]
            perfs = agg.encoder_perfs[wkey]
            perfs /= perfs.max()

            W = np.abs(agg.encoder_weights[wkey])
            for k,aprop in enumerate(aprops):
                W[:, :, k] *= perfs

            Wmean = W.mean(axis=0)
            all_weights.append(Wmean)

        all_weights = np.array(all_weights)
        w_by_decomp[decomp] = all_weights.mean(axis=0)

    aprops_to_display = ['sal', 'maxAmp', 'meanspect', 'entropyspect', 'q2', 'entropytime']

    figsize = (24, 7)
    fig = plt.figure(figsize=figsize)
    clrs = [ACOUSTIC_FEATURE_COLORS[aprop] for aprop in aprops_to_display]

    ax = plt.subplot(1, 2, 1)
    for aprop in aprops_to_display:
        m = aprops.index(aprop)
        plt.plot(freqs, w_by_decomp['self_locked'][:, m], '-', c=ACOUSTIC_FEATURE_COLORS[aprop], linewidth=7.0, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Weight Magnitude')
    plt.axis('tight')
    plt.title('Weight Importance (LFP PSD)')
    leg = custom_legend(clrs, aprops_to_display)
    plt.legend(handles=leg)

    ax = plt.subplot(1, 2, 2)
    for aprop in aprops_to_display:
        m = aprops.index(aprop)
        plt.plot(freqs, w_by_decomp['self_spike_psd'][:, m], '-', c=ACOUSTIC_FEATURE_COLORS[aprop], linewidth=7.0, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mean Weight Magnitude')
    plt.axis('tight')
    plt.title('Weight Importance (Spike PSD)')
    leg = custom_legend(clrs, aprops_to_display)
    plt.legend(handles=leg)

    fname = os.path.join(get_this_dir(), 'encoder_weights_by_freq.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)

    draw_encoder_perfs(agg)
    # draw_encoder_weights(agg)

    """
    for (bird,block,segment,hemi,decomp),gdf in agg.df.groupby(['bird', 'block', 'segment','hemi','decomp']):

        assert len(gdf) == 1

        if decomp in ['self_spike_rate', 'self+cross_locked']:
            continue

        # get segment/decomp props
        wkey = gdf.wkey.values[0]
        iindex = gdf.iindex.values[0]

        # get encoder performances
        encoder_perfs = agg.encoder_perfs[wkey]

        # get decoder weights
        decoder_weights = agg.decoder_weights[wkey]
        decoder_perfs = agg.decoder_perfs[wkey]

        fig = plt.figure(figsize=(24, 13))
        plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.15, wspace=0.15)
        ncols = 4
        nrows = 3

        plt.subplot(nrows, ncols, 1)
        plt.imshow(encoder_perfs, interpolation='nearest', aspect='auto', vmin=0, cmap=magma)
        plt.colorbar()
        plt.xticks()
        plt.title('Encoder Perf')

        for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
            dW = decoder_weights[:, :, k]
            plt.subplot(nrows, ncols, k+2)
            absmax = np.abs(dW).max()
            plt.imshow(dW, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
            plt.colorbar()
            plt.title('%s decoder: %0.2f' % (aprop, decoder_perfs[k]))

        seg_fname = '%s_%s_%s_%s_%s' % (bird, block, segment, hemi, decomp)
        plt.suptitle(seg_fname)
        fname = os.path.join(fig_dir, '%s.png' % seg_fname)

        plt.savefig(fname)
        plt.close('all')
    """

if __name__ == '__main__':
    set_font()
    draw_figures()
