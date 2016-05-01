import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from lasp.plots import multi_plot, custom_legend
from lasp.colormaps import magma

from DecodingRhythms.utils import set_font, clean_region
from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder
from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import ALL_ACOUSTIC_PROPS


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
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
        ii = (gdf.decomp == 'self_locked')
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
    aprops_to_show = ['sal', 'q2', 'maxAmp', 'entropytime']
    electrode_props = list()
    for f in sorted(df.freq.unique()):
        i = df.freq == f

        all_cols = dict()
        all_cols['dm'] = df.dist_midline[i].values
        all_cols['dl'] = df.dist_l2a[i].values
        all_cols['r2'] = df.encoder_r2[i].values
        for aprop in aprops_to_show:
            eperf = df[i]['eperf_ind_%s' % aprop].values
            ew = df[i]['eweight_ind_%s' % aprop].values
            all_cols[aprop] = (eperf / eperf.max()) * ew

        df_f = pd.DataFrame(all_cols)
        gi = (df_f.dm < 2.5) & ~np.isnan(df_f.dm) & ~np.isnan(df_f.dl) & (df_f.r2 > 0)
        df_f = df_f[gi]
        electrode_props.append({'f':f, 'df':df_f})

    def _plot_map(_pdata, _ax, _prop, _cmap, _maxval=None, _bgcolor=None):
        if _bgcolor is not None:
            _ax.set_axis_bgcolor(_bgcolor)
        _pval = _pdata['df'][_prop].values
        _x = _pdata['df'].dm.values
        _y = _pdata['df'].dl.values

        plt.sca(_ax)
        if _maxval is not None:
            _clrs = _cmap(_pval / _maxval)

        # _ax.set_axis_bgcolor('black')
        for k,(_xx,_yy) in enumerate(zip(_x, _y)):
            plt.plot(_xx, _yy, 'o', c=_clrs[k], markersize=8, alpha=0.9)

        plt.title('f=%d' % _pdata['f'])
        # plt.colorbar()

    absmax = dict()
    for aprop in aprops_to_show:
        absmax[aprop] = np.abs(df['eweight_ind_%s' % aprop]).max()

    def rb_cmap(x):
        assert np.abs(x).max() <= 1
        _rgb = np.ones([len(x), 3])
        _pos = x >= 0
        _neg = x < 0

        _rgb[_pos, 0] = x[_pos]
        _rgb[_neg, 2] = np.abs(x[_neg])

        return _rgb
    
    def _plot_r2_map(_pdata, _ax): _plot_map(_pdata, _ax, 'r2', magma, _maxval=0.30, _bgcolor='black')
    multi_plot(electrode_props, _plot_r2_map, nrows=3, ncols=4, figsize=(23, 13))
    plt.suptitle('Encoder Performance (R2)')

    def _plot_sal_map(_pdata, _ax): _plot_map(_pdata, _ax, 'sal', rb_cmap, _maxval=absmax['sal'])
    multi_plot(electrode_props, _plot_sal_map, nrows=3, ncols=4, figsize=(23, 13))
    plt.suptitle('Saliency Encoder Weights')

    def _plot_q2_map(_pdata, _ax):
        _plot_map(_pdata, _ax, 'q2', rb_cmap, _maxval=absmax['q2'])
    multi_plot(electrode_props, _plot_q2_map, nrows=3, ncols=4, figsize=(23, 13))
    plt.suptitle('Q2 Encoder Weights')

    def _plot_maxAmp_map(_pdata, _ax):
        _plot_map(_pdata, _ax, 'maxAmp', rb_cmap, _maxval=absmax['maxAmp'])
    multi_plot(electrode_props, _plot_maxAmp_map, nrows=3, ncols=4, figsize=(23, 13))
    plt.suptitle('maxAmp Encoder Weights')

    def _plot_entropytime_map(_pdata, _ax):
        _plot_map(_pdata, _ax, 'entropytime', rb_cmap, _maxval=absmax['entropytime'])
    multi_plot(electrode_props, _plot_entropytime_map, nrows=3, ncols=4, figsize=(23, 13))
    plt.suptitle('entropytime Encoder Weights')

    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)

    plot_maps(agg)
    # plot_raw_dists()


if __name__ == '__main__':
    # set_font()
    draw_figures()
