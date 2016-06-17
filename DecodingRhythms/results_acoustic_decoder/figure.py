import os
from copy import deepcopy, copy

import h5py
import numpy as np
import operator
import pandas as pd
import matplotlib.pyplot as plt

from lasp.plots import custom_legend, compute_mean_from_scatter

from DecodingRhythms.utils import set_font, get_this_dir, clean_region, COLOR_RED_SPIKE_RATE, COLOR_BLUE_LFP, \
    COLOR_PURPLE_LFP_CROSS, COLOR_CRIMSON_SPIKE_SYNC, get_e2e_dists

from zeebeez.aggregators.acoustic_encoder_decoder import AcousticEncoderDecoderAggregator
from zeebeez.utils import REDUCED_ACOUSTIC_PROPS, ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT, \
    ACOUSTIC_FEATURE_COLORS, USED_ACOUSTIC_PROPS, ACOUSTIC_PROP_COLORS_BY_TYPE, ACOUSTIC_PROP_NAMES


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_Site1_Call1_L_full_psds.h5')
    lags = hf.attrs['lags']
    freqs = hf.attrs['freqs']
    hf.close()
    nlags = len(lags)

    return freqs,lags


def export_decoder_datasets_for_glm(agg, data_dir='/auto/tdrive/mschachter/data'):

    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_Site4_Call1_L_full_psds.h5', 'r')
    freqs = hf.attrs['freqs']
    hf.close()

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'decomp':list(), 'site':list(),
            'aprop':list(), 'r2':list()}

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi', 'decomp'])
    for (bird,block,seg,hemi,decomp),gdf in g:

        site = '%s_%s_%s_%s' % (bird, block, seg, hemi)

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        dperf = agg.decoder_perfs[wkey]

        for k,aprop in enumerate(USED_ACOUSTIC_PROPS):

            r2 = dperf[k]

            data['bird'].append(bird)
            data['block'].append(block)
            data['segment'].append(seg)
            data['hemi'].append(hemi)
            data['decomp'].append(decomp)
            data['site'].append(site)
            data['aprop'].append(aprop)
            data['r2'].append(r2)

    df = pd.DataFrame(data)

    print 'decomps=',df.decomp.unique()
    df.to_csv(os.path.join(data_dir, 'aggregate', 'decoder_perfs_for_glm.csv'), header=True, index=False)


def export_weight_ds(agg, data_dir='/auto/tdrive/mschachter/data'):

    freqs,lags = get_freqs_and_lags()

    assert isinstance(agg, AcousticEncoderDecoderAggregator)
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    data = {'bird': list(), 'block': list(), 'segment': list(), 'hemi': list(),
            'electrode': list(), 'reg': list(), 'dm': list(), 'dl': list(),
            'aprop': list(), 'r2': list(), 'f':list(), 'w':list()}

    aprops = USED_ACOUSTIC_PROPS
    nprops = len(aprops)

    i = agg.df.decomp == 'full_psds'
    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird, block, segment, hemi), gdf in g:

        assert len(gdf) == 1
        wkey = gdf.wkey.values[0]
        index2electrode = agg.index2electrode[wkey]
        nelectrodes = len(index2electrode)
        nfreqs = len(freqs)

        dweights = agg.decoder_weights[wkey]
        assert dweights.shape == (nelectrodes, nfreqs, nprops), "dweights.shape=%s" % str(dweights.shape)

        dperfs = agg.decoder_perfs[wkey]
        assert len(dperfs) == nprops

        for k,e in enumerate(index2electrode):

            ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert ei.sum() == 1
            reg = clean_region(edata.region[ei].values[0])
            dist_l2a = edata.dist_l2a[ei].values[0]
            dist_midline = edata.dist_midline[ei].values[0]

            if bird == 'GreBlu9508M':
                dist_l2a *= 4

            for j,f in enumerate(freqs):
                for m,aprop in enumerate(aprops):

                    data['bird'].append(bird)
                    data['block'].append(block)
                    data['segment'].append(segment)
                    data['hemi'].append(hemi)
                    data['electrode'].append(e)
                    data['reg'].append(reg)
                    data['dm'].append(dist_midline)
                    data['dl'].append(dist_l2a)
                    data['aprop'].append(aprop)
                    data['r2'].append(dperfs[m])
                    data['f'].append(int(f))
                    data['w'].append(dweights[k, j, m])

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(data_dir, 'aggregate', 'decoder_weights_for_glm.csv'), header=True, index=False)


def draw_decoder_perf_barplots(data_dir='/auto/tdrive/mschachter/data', show_all=True):

    aprops_to_display = list(USED_ACOUSTIC_PROPS)

    if not show_all:
        decomps = ['spike_rate', 'full_psds']
        sub_names = ['Spike Rate', 'LFP PSD']
        sub_clrs = [COLOR_RED_SPIKE_RATE, COLOR_BLUE_LFP]
    else:
        decomps = ['spike_rate', 'full_psds', 'spike_rate+spike_sync', 'full_psds+full_cfs']
        sub_names = ['Spike Rate', 'LFP PSD', 'Spike Rate + Sync', 'LFP PSD + CFs']
        sub_clrs = [COLOR_RED_SPIKE_RATE, COLOR_BLUE_LFP, COLOR_CRIMSON_SPIKE_SYNC, COLOR_PURPLE_LFP_CROSS]

    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'decoder_perfs_for_glm.csv'))
    bprop_data = list()

    for aprop in aprops_to_display:
        bd = dict()
        for decomp in decomps:
            i = (df_me.decomp == decomp) & (df_me.aprop == aprop)
            perfs = df_me.r2[i].values
            bd[decomp] = perfs
        bprop_data.append({'bd':bd, 'lfp_mean':bd['full_psds'].mean(), 'aprop':aprop})

    bprop_data.sort(key=operator.itemgetter('lfp_mean'), reverse=True)

    lfp_r2 = [bdict['bd']['full_psds'].mean() for bdict in bprop_data]
    lfp_r2_std = [bdict['bd']['full_psds'].std(ddof=1) for bdict in bprop_data]

    spike_r2 = [bdict['bd']['spike_rate'].mean() for bdict in bprop_data]
    spike_r2_std = [bdict['bd']['spike_rate'].std(ddof=1) for bdict in bprop_data]

    if show_all:
        pairwise_r2 = [bdict['bd']['full_psds+full_cfs'].mean() for bdict in bprop_data]
        pairwise_r2_std = [bdict['bd']['full_psds+full_cfs'].std(ddof=1) for bdict in bprop_data]
        spike_sync_r2 = [bdict['bd']['spike_rate+spike_sync'].mean() for bdict in bprop_data]
        spike_sync_r2_std = [bdict['bd']['spike_rate+spike_sync'].std(ddof=1) for bdict in bprop_data]

    aprops_xticks = [ACOUSTIC_PROP_NAMES[bdict['aprop']] for bdict in bprop_data]

    figsize = (23, 7.)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.05, right=0.99, hspace=0.20, wspace=0.20)

    bar_width = 0.4
    if show_all:
        bar_width = 0.2

    bar_data = [(spike_r2, spike_r2_std), (lfp_r2, lfp_r2_std)]
    if len(decomps) == 4:
        bar_data.append( (spike_sync_r2, spike_sync_r2_std) )
        bar_data.append( (pairwise_r2, pairwise_r2_std) )

    bar_x = np.arange(len(lfp_r2))
    for k,(br2,bstd) in enumerate(bar_data):
        bx = bar_x + bar_width*k
        plt.bar(bx, br2, yerr=bstd, width=bar_width, color=sub_clrs[k], alpha=0.9, ecolor='k')

    plt.ylabel('Decoder R2')
    plt.xticks(bar_x+0.45, aprops_xticks, rotation=90, fontsize=12)

    leg = custom_legend(sub_clrs, sub_names)
    plt.legend(handles=leg, loc='upper right')
    plt.axis('tight')
    plt.xlim(-0.5, bar_x.max() + 1)
    plt.ylim(0, 0.5)

    fname = os.path.join(get_this_dir(), 'decoder_perf_barplots.svg')
    if show_all:
        fname = os.path.join(get_this_dir(), 'decoder_perf_barplots_all.svg')

    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_pairwise_weights_vs_dist(agg):

    wdf = export_pairwise_decoder_weights(agg)

    r2_thresh = 0.20
    # aprops = ALL_ACOUSTIC_PROPS
    aprops = ['meanspect', 'stdspect', 'sal', 'maxAmp']
    aprop_clrs = {'meanspect':'#FF8000', 'stdspect':'#FFBF00', 'sal':'#088A08', 'maxAmp':'k'}

    print wdf.decomp.unique()

    # get scatter data for weights vs distance
    scatter_data = dict()
    for decomp in ['full_psds+full_cfs', 'spike_rate+spike_sync']:
        i = wdf.decomp == decomp
        assert i.sum() > 0

        df = wdf[i]
        for n,aprop in enumerate(aprops):
            ii = (df.r2 > r2_thresh) & (df.aprop == aprop) & ~np.isnan(df.dist) & ~np.isinf(df.dist)
            d = df.dist[ii].values
            w = df.w[ii].values
            scatter_data[(decomp, aprop)] = (d, w)

    decomp_labels = {'full_psds+full_cfs':'LFP Pairwise Correlations', 'spike_rate+spike_sync':'Spike Synchrony'}

    figsize = (14, 5)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.10, right=0.98)

    for k,decomp in enumerate(['full_psds+full_cfs', 'spike_rate+spike_sync']):
        ax = plt.subplot(1, 2, k+1)
        for aprop in aprops:
            d,w = scatter_data[(decomp, aprop)]

            wz = w**2
            wz /= wz.std(ddof=1)

            xcenter, ymean, yerr, ymean_cs = compute_mean_from_scatter(d, wz, bins=5, num_smooth_points=300)
            # plt.plot(xcenter, ymean, '-', c=aprop_clrs[aprop], linewidth=7.0, alpha=0.7)
            plt.errorbar(xcenter, ymean, linestyle='-', yerr=yerr, c=aprop_clrs[aprop], linewidth=9.0, alpha=0.7,
                         elinewidth=8., ecolor='#d8d8d8', capsize=0.)
            plt.axis('tight')

        plt.ylabel('Decoder Weight Effect')
        plt.xlabel('Pairwise Distance (mm)')
        plt.title(decomp_labels[decomp])

        aprop_lbls = [ACOUSTIC_PROP_NAMES[aprop] for aprop in aprops]
        aclrs = [aprop_clrs[aprop] for aprop in aprops]
        leg = custom_legend(aclrs, aprop_lbls)
        plt.legend(handles=leg, loc='lower left')

    fname = os.path.join(get_this_dir(), 'pairwise_decoder_effect_vs_dist.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()



def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'acoustic_encoder_decoder.h5')
    agg = AcousticEncoderDecoderAggregator.load(agg_file)

    # ###### these two functions write a csv file for decoder weights and draw barplots for decoder performance
    export_decoder_datasets_for_glm(agg)
    draw_decoder_perf_barplots(show_all=True)

    # ###### these two functions draw the relationship between pairwise decoder weights and distance
    # draw_pairwise_weights_vs_dist(agg)

    # export_weight_ds(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()
