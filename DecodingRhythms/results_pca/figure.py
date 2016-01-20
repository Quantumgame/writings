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

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)

    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))

    # aprops_to_display = ['category', 'maxAmp', 'sal', 'meantime', 'entropytime', 'meanspect', 'q1', 'q2', 'q3',
    #                      'entropyspect']
    aprops_to_display = ['category', 'maxAmp', 'sal', 'q1', 'q2', 'q3']

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
        plt.errorbar(_x, _lfp_mean, yerr=_lfp_std, linewidth=6.0, alpha=0.8, c=COLOR_BLUE_LFP, ecolor='k', elinewidth=2.0)
        plt.errorbar(_x+0.35, _spike_mean, yerr=_spike_std, linewidth=6.0, alpha=0.8, c=COLOR_YELLOW_SPIKE, ecolor='k', elinewidth=2.0)
        plt.ylabel('Performance Fraction')
        plt.xlabel('Frequency Resolution (# of PCs)')
        plt.xticks(_x, ['%d' % d for d in _x])
        plt.axis('tight')
        plt.ylim(0.1, 1.1)
        plt.title(_pdata['aprop'])
        leg = custom_legend([COLOR_BLUE_LFP, COLOR_YELLOW_SPIKE], ['LFP', 'Spike PSD'])
        plt.legend(handles=leg, loc='lower right', fontsize='small')

    multi_plot(plist, _plot_perf_frac, nrows=2, ncols=3, hspace=0.30, wspace=0.30, figsize=(24, 8))
    fname = os.path.join(get_this_dir(), 'perf_fracs.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':
    set_font()
    draw_figures()
    plt.show()