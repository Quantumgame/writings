import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from DecodingRhythms.utils import COLOR_BLUE_LFP, get_this_dir, set_font
from lasp.plots import grouped_boxplot, custom_legend

from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    # do PCA to look at weight variation
    pp_file = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    pt = PairwiseCFTransform.load(pp_file)
    psds = deepcopy(pt.psds)
    psum = psds.sum(axis=1)
    i = (psum > 0) & ~np.isnan(psum)
    psds = psds[i]
    pt.log_transform(psds)

    pca = PCA()
    pca.fit(psds)
    print 'Explained variance: ',pca.explained_variance_ratio_
    pcs = pca.components_
    print 'pcs.shape=',pcs.shape

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)

    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))

    aprops_to_display = ['category', 'maxAmp', 'sal', 'meanspect', 'q1', 'q2', 'q3',
                         'entropyspect', 'meantime', 'entropytime']

    df0 = df_me[df_me.band == 0]

    bp_data_lfp = dict()
    bp_data_spike = dict()
    for aprop in aprops_to_display:

        perfs = df0['perf_%s_lfp' % aprop].values
        perfs_1d = df0['perf_%s_locked_pca_1' % aprop].values
        perfs_2d = df0['perf_%s_locked_pca_2' % aprop].values
        bp_data_lfp[aprop] = [perfs, perfs_2d, perfs_1d]

        perfs = df0['perf_%s_spike' % aprop].values
        perfs_1d = df0['perf_%s_spike_psd_pca_1' % aprop].values
        perfs_2d = df0['perf_%s_spike_psd_pca_2' % aprop].values
        bp_data_spike[aprop] = [perfs, perfs_2d, perfs_1d]

    figsize = (24, 12)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    gs = plt.GridSpec(2, 100)

    ax = plt.subplot(gs[0, :30])
    x = np.arange(len(agg.freqs))
    plt.bar(left=x, height=pca.explained_variance_ratio_, color='gray')
    plt.xlabel('PC')
    plt.ylabel('Explained Variance')

    ax = plt.subplot(gs[1, :30])
    absmax = np.abs(pcs).max()
    plt.axhline(0, c='k')
    clrs = ['k', '#005289', '#00B1E1']
    for k,pc in enumerate(pcs[:3]):
        plt.plot(agg.freqs, pc, '-', c=clrs[k], linewidth=3.0, alpha=0.7)
    plt.ylim(-absmax, absmax)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Principal Component')
    leg = custom_legend(clrs, ['PC1', 'PC2', 'PC3'])
    plt.legend(handles=leg)
    plt.axis('tight')

    ax = plt.subplot(gs[0, 35:])
    grouped_boxplot(bp_data_lfp, group_names=aprops_to_display, subgroup_names=['LFP', 'PCA (2D)', 'PCA (1D)'],
                    subgroup_colors=[COLOR_BLUE_LFP, 'gray', 'white'], box_spacing=1.5,
                    ax=ax)
    plt.xlabel('Acoustic Feature')
    plt.ylabel('Decoder R2')

    ax = plt.subplot(gs[1, 35:])
    grouped_boxplot(bp_data_spike, group_names=aprops_to_display, subgroup_names=['Spike PSD', 'PCA (2D)', 'PCA (1D)'],
                    subgroup_colors=[COLOR_BLUE_LFP, 'gray', 'white'], box_spacing=1.5,
                    ax=ax)
    plt.xlabel('Acoustic Feature')
    plt.ylabel('Decoder R2')

    fname = os.path.join(get_this_dir(), 'perf_boxplots.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':
    set_font()
    draw_figures()
    plt.show()