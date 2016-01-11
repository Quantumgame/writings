import os
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from DecodingRhythms.utils import COLOR_BLUE_LFP, get_this_dir, set_font, COLOR_YELLOW_SPIKE
from lasp.plots import grouped_boxplot, custom_legend, multi_plot
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

    aprops_to_display = ['category', 'maxAmp', 'sal', 'meantime', 'entropytime', 'meanspect', 'q1', 'q2', 'q3',
                         'entropyspect']

    df0 = df_me[(df_me.band == 0) & (df_me.bird != 'BlaBro09xxF')]

    perf_frac_vs_pc_lfp = dict()
    perf_frac_vs_pc_spike = dict()

    for aprop in aprops_to_display:
        perf_frac_vs_pc_lfp[aprop] = list()
        perf_frac_vs_pc_spike[aprop] = list()

    g = df0.groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird,block,segment,hemi),gdf in g:
        assert len(gdf) == 1

        for aprop in aprops_to_display:
            
            full_perf_lfp = gdf['perf_%s_lfp' % aprop].values[0]
            pfrac = list()
            for ncomp in range(1, 12):
                ncomp_perf = gdf['perf_%s_locked_pca_%d' % (aprop, ncomp)].values[0]
                pfrac.append(ncomp_perf / full_perf_lfp)
            perf_frac_vs_pc_lfp[aprop].append(np.array(pfrac))
            
            full_perf_spike = gdf['perf_%s_spike' % aprop].values[0]
            pfrac = list()
            for ncomp in range(1, 12):
                ncomp_perf = gdf['perf_%s_spike_psd_pca_%d' % (aprop, ncomp)].values[0]
                pfrac.append(ncomp_perf / full_perf_spike)
            perf_frac_vs_pc_spike[aprop].append(np.array(pfrac))

    plist = [{'aprop':aprop, 'lfp':perf_frac_vs_pc_lfp[aprop], 'spike':perf_frac_vs_pc_spike[aprop]} for aprop in aprops_to_display]

    def _plot_perf_frac(_pdata, _ax):
        plt.sca(_ax)
        _lfp = np.array(_pdata['lfp'])
        _spike = np.array(_pdata['spike'])
        _lfp_mean = _lfp.mean(axis=0)
        _lfp_std = _lfp.std(axis=0, ddof=1)
        _spike_mean = _spike.mean(axis=0)
        _spike_std = _spike.std(axis=0, ddof=1)

        plt.axhline(1.0, c='k', alpha=0.7, linestyle='dashed', linewidth=3.0)
        _x = np.arange(1, 12)
        plt.errorbar(_x, _lfp_mean, yerr=_lfp_std, linewidth=6.0, alpha=0.8, c=COLOR_BLUE_LFP)
        plt.errorbar(_x+0.35, _spike_mean, yerr=_spike_std, linewidth=6.0, alpha=0.8, c=COLOR_YELLOW_SPIKE)
        plt.ylabel('Performance Fraction')
        plt.xlabel('Frequency Resolution (# of PCs)')
        plt.xticks(_x, ['%d' % d for d in _x])
        plt.axis('tight')
        plt.ylim(0.1, 1.1)
        plt.title(_pdata['aprop'])
        leg = custom_legend([COLOR_BLUE_LFP, COLOR_YELLOW_SPIKE], ['LFP', 'Spike PSD'])
        plt.legend(handles=leg, loc='lower right', fontsize='small')

    multi_plot(plist, _plot_perf_frac, nrows=2, ncols=5, hspace=0.30, wspace=0.30, figsize=(24, 8))
    fname = os.path.join(get_this_dir(), 'perf_fracs.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    bp_data_lfp = dict()
    bp_data_spike = dict()
    for aprop in aprops_to_display:

        perfs = df0['perf_%s_lfp' % aprop].values
        perfs_1d = df0['perf_%s_locked_pca_1' % aprop].values
        perfs_8d = df0['perf_%s_locked_pca_8' % aprop].values
        bp_data_lfp[aprop] = [perfs, perfs_8d, perfs_1d]

        perfs = df0['perf_%s_spike' % aprop].values
        perfs_1d = df0['perf_%s_spike_psd_pca_1' % aprop].values
        perfs_8d = df0['perf_%s_spike_psd_pca_8' % aprop].values
        bp_data_spike[aprop] = [perfs, perfs_8d, perfs_1d]

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
    grouped_boxplot(bp_data_lfp, group_names=aprops_to_display, subgroup_names=['LFP', 'PCA (8D)', 'PCA (1D)'],
                    subgroup_colors=[COLOR_BLUE_LFP, 'gray', 'white'], box_spacing=1.5,
                    ax=ax)
    plt.xlabel('Acoustic Feature')
    plt.ylabel('Decoder R2')

    ax = plt.subplot(gs[1, 35:])
    grouped_boxplot(bp_data_spike, group_names=aprops_to_display, subgroup_names=['Spike PSD', 'PCA (8D)', 'PCA (1D)'],
                    subgroup_colors=[COLOR_YELLOW_SPIKE, 'gray', 'white'], box_spacing=1.5,
                    ax=ax)
    plt.xlabel('Acoustic Feature')
    plt.ylabel('Decoder R2')

    fname = os.path.join(get_this_dir(), 'perf_boxplots.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':
    set_font()
    draw_figures()
    plt.show()