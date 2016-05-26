import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from lasp.plots import multi_plot, custom_legend
from lasp.colormaps import magma

from DecodingRhythms.utils import set_font, clean_region, get_this_dir
from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder
from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import ALL_ACOUSTIC_PROPS


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_Site4_Call1_L_full_psds.h5')
    lags = hf.attrs['lags']
    freqs = hf.attrs['freqs']
    hf.close()

    return freqs,lags


def export_perf_and_weight_and_loc_data(agg, data_dir='/auto/tdrive/mschachter/data'):

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    freqs,lags = get_freqs_and_lags()

    font = {'family':'normal', 'weight':'bold', 'size':10}
    plt.matplotlib.rc('font', **font)

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
            'electrode':list(), 'region':list(), 'freq':list(), 'dist_midline':list(), 'dist_l2a':list(),
            'encoder_r2':list()}

    for k,aprop in enumerate(ALL_ACOUSTIC_PROPS):
        data['eperf_ind_%s' % aprop] = list()
        data['eweight_ind_%s' % aprop] = list()
        data['dweight_%s' % aprop] = list()

    for (bird,block,segment,hemi),gdf in agg.df.groupby(['bird', 'block', 'segment', 'hemi']):
        bstr = '%s_%s_%s_%s' % (bird,hemi,block,segment)
        ii = (gdf.decomp == 'full_psds')
        assert ii.sum() == 1
        wkey = gdf[ii]['wkey'].values[0]
        iindex = gdf[ii]['iindex'].values[0]

        index2electrode = agg.index2electrode[iindex]

        lfp_eperf = agg.encoder_perfs[wkey]
        lfp_eperf_ind = agg.encoder_perfs_ind[wkey]
        lfp_eweights_ind = agg.encoder_weights_ind[wkey]

        lfp_decoder_weights = agg.decoder_weights[wkey]
        lfp_decoder_perfs = agg.decoder_perfs[wkey]

        for k,e in enumerate(index2electrode):

            ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert ei.sum() == 1
            reg = clean_region(edata.region[ei].values[0])
            dist_l2a = edata.dist_l2a[ei].values[0]
            dist_midline = edata.dist_midline[ei].values[0]

            if bird == 'GreBlu9508M':
                dist_l2a *= 4

            for j,f in enumerate(freqs):

                enc_r2 = lfp_eperf[k, j]

                data['bird'].append(bird)
                data['block'].append(block)
                data['segment'].append(segment)
                data['hemi'].append(hemi)
                data['electrode'].append(e)
                data['region'].append(reg)
                data['freq'].append(int(f))
                data['dist_midline'].append(dist_midline)
                data['dist_l2a'].append(dist_l2a)
                data['encoder_r2'].append(enc_r2)

                for m,aprop in enumerate(ALL_ACOUSTIC_PROPS):
                    eperf_ind = lfp_eperf_ind[k, j, m]
                    eweight_ind = lfp_eweights_ind[k, j, m]
                    dweight = lfp_decoder_weights[k, j, m]
                    data['eperf_ind_%s' % aprop].append(eperf_ind)
                    data['eweight_ind_%s' % aprop].append(eweight_ind)
                    data['dweight_%s' % aprop].append(dweight)

    return pd.DataFrame(data)


def plot_raw_dists(data_dir='/auto/tdrive/mschachter/data'):
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    i = edata.bird != 'BlaBro09xxF'
    edata = edata[i]
    birds = edata.bird.unique()

    clrs = ['r', 'g', 'b']

    fig = plt.figure()
    fig.subplots_adjust(top=0.99, bottom=0.01, right=0.99, left=0.01, hspace=0, wspace=0)

    for k,b in enumerate(birds):
        i = (edata.bird == b) & ~np.isnan(edata.dist_midline) & ~np.isnan(edata.dist_l2a)
        x = edata[i].dist_midline.values
        y = edata[i].dist_l2a.values
        if b == 'GreBlu9508M':
            y *= 4
        reg = edata[i].region.values

        for xx,yy,r in zip(x, y, reg):
            plt.plot(xx, yy, 'o', markersize=8, c=clrs[k], alpha=0.4)
            plt.text(xx, yy, r, fontsize=8)
        plt.axis('tight')
        plt.xlabel('Dist from Midline (mm)')
        plt.ylabel('Dist from LH (mm)')

    leg = custom_legend(clrs, birds)
    plt.legend(handles=leg)
    plt.show()


def plot_maps(agg):

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
            'electrode':list(), 'region':list(), 'freq':list(), 'dist_midline':list(), 'dist_l2a':list(),
            'encoder_r2':list()}

    df = export_perf_and_weight_and_loc_data(agg)

    print 'dist_l2a: min=%f, max=%f' % (df.dist_l2a.min(), df.dist_l2a.max())
    print 'dist_midline: min=%f, max=%f' % (df.dist_midline.min(), df.dist_midline.max())

    gx,gy = np.linspace(0.0, 2.5), np.linspace(-1.45, 1.05, 100)
    gridX, gridY = np.meshgrid(gx, gy)

    # encoder performance maps
    aprops_to_show = ALL_ACOUSTIC_PROPS
    electrode_props = list()
    for f in sorted(df.freq.unique()):
        i = df.freq == f

        all_cols = dict()
        all_cols['dm'] = df.dist_midline[i].values
        all_cols['dl'] = df.dist_l2a[i].values
        all_cols['r2'] = df.encoder_r2[i].values
        all_cols['reg'] = df.region[i].values
        for aprop in aprops_to_show:
            eperf = df[i]['eperf_ind_%s' % aprop].values
            ew = df[i]['eweight_ind_%s' % aprop].values
            all_cols[aprop] = ew
            all_cols['%s_perf' % aprop] = eperf

        df_f = pd.DataFrame(all_cols)
        gi = (df_f.dm < 2.5) & ~np.isnan(df_f.dm) & ~np.isnan(df_f.dl) & (df_f.r2 > 0)
        df_f = df_f[gi]
        electrode_props.append({'f':int(f), 'df':df_f})

    def _plot_map(_pdata, _ax, _prop, _cmap, _maxval, _bgcolor=None, _perf_alpha=False, _plot_region=False, _msize=8):
        if _bgcolor is not None:
            _ax.set_axis_bgcolor(_bgcolor)
        _pval = _pdata['df'][_prop].values
        _x = _pdata['df'].dm.values
        _y = _pdata['df'].dl.values
        _regs = _pdata['df']['reg'].values

        plt.sca(_ax)
        _alpha = np.ones([len(_pval)])
        if _perf_alpha:
            _alpha = _pdata['df']['%s_perf' % _prop].values
            _alpha /= _alpha.max()
            _alpha[_alpha > 0.9] = 1.
            _clrs = _cmap(_pval / _maxval)
        else:
            _clrs = _cmap(_pval / _maxval)

        for k,(_xx,_yy) in enumerate(zip(_x, _y)):
            plt.plot(_xx, _yy, 'o', c=_clrs[k], alpha=_alpha[k], markersize=_msize)
            # plt.text(_xx, _yy, _regs[k], fontsize=8, color='w', alpha=0.7)

        plt.title('f=%d' % _pdata['f'])

    absmax = dict()
    for aprop in aprops_to_show:
        absmax[aprop] = np.abs(df['eweight_ind_%s' % aprop]).max()

    def rb_cmap(x):
        assert np.abs(x).max() <= 1
        _rgb = np.zeros([len(x), 3])
        _pos = x >= 0
        _neg = x < 0

        _rgb[_pos, 0] = x[_pos]
        _rgb[_neg, 2] = np.abs(x[_neg])

        return _rgb

    max_r2 = 0.40
    """
    def _plot_r2_map(_pdata, _ax): _plot_map(_pdata, _ax, 'r2', magma, _maxval=max_r2, _bgcolor='black')
    multi_plot(electrode_props, _plot_r2_map, nrows=3, ncols=4, figsize=(23, 13))
    plt.suptitle('Encoder Performance (R2)')
    plt.show()

    fname = os.path.join(get_this_dir(), 'r2_map_allfreq.png')
    plt.savefig(fname, facecolor='w', edgecolor='none')
    """

    """
    for aprop in aprops_to_show:
        def _plot_aprop_map(_pdata, _ax): _plot_map(_pdata, _ax, aprop, rb_cmap, _maxval=absmax[aprop], _perf_alpha=True)
        multi_plot(electrode_props, _plot_aprop_map, nrows=3, ncols=4, figsize=(23, 13))
        plt.suptitle('%s Univariate Encoder Weights' % aprop)
        fname = os.path.join(get_this_dir(), 'map_allfreq_%s.png' % aprop)
        plt.savefig(fname, facecolor='w', edgecolor='none')
    """

    """
    edict = [e for e in electrode_props if e['f'] == 33][0]

    # make a plot for just saliency at 33Hz
    aprops = {'sal':'Saliency', 'meanspect':'Spectral Mean', 'maxAmp':'Maximum Amplitude'}

    set_font()
    for aprop,aname in aprops.items():
        figsize = (12, 10)
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        ax.set_axis_bgcolor('black')
        _plot_map(edict, ax, aprop, rb_cmap, _maxval=absmax[aprop], _perf_alpha=True, _msize=12)
        plt.xlabel('Distance from midline (mm)')
        plt.ylabel('Distance from L2A (mm)')
        plt.title('%s Encoder Weights (33Hz)' % aname)

        fname = os.path.join(get_this_dir(), '%s_encoder_weights.svg' % aprop)
        plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()
    """

    """
    # make a plot just for r2 at 33 Hz
    set_font()
    edict = [e for e in electrode_props if e['f'] == 33][0]
    figsize = (23, 10)
    plt.figure(figsize=figsize)
    gs = plt.GridSpec(1, 100)

    ax = plt.subplot(gs[50:])
    ax.set_axis_bgcolor('black')
    pval = edict['df']['r2'].values
    pval /= max_r2
    regs = edict['df']['reg'].values
    x = edict['df'].dm.values
    y = edict['df'].dl.values
    for k, (_xx, _yy) in enumerate(zip(x, y)):
        plt.plot(_xx, _yy, 'o', c=magma(pval[k]), alpha=0.9, markersize=8)
        plt.text(_xx, _yy, regs[k], fontsize=8, color='w', alpha=0.7)

    plt.xlabel('Distance from LH (mm)')
    plt.ylabel('Distance from L2A (mm)')
    """

    fname = os.path.join(get_this_dir(), 'r2_map.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    # plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)

    plot_maps(agg)
    # plot_raw_dists()


if __name__ == '__main__':
    # set_font()
    draw_figures()
