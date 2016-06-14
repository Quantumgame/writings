import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lasp.plots import multi_plot, custom_legend
from lasp.colormaps import magma

from DecodingRhythms.utils import set_font, clean_region, get_this_dir
from zeebeez.aggregators.single_electrode_decoder import SingleElectrodeDecoderAggregator
from zeebeez.utils import ALL_ACOUSTIC_PROPS, ACOUSTIC_PROP_NAMES


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_Site4_Call1_L_full_psds.h5')
    lags = hf.attrs['lags']
    freqs = hf.attrs['freqs']
    hf.close()

    return freqs,lags


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


def plot_maps(agg, data_dir='/auto/tdrive/mschachter/data'):

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
            'electrode':list(), 'reg':list(), 'dm':list(), 'dl':list(),
            'aprop':list(), 'r2':list()}

    df = agg.df
    # encoder performance maps
    aprops_to_show = ['stdtime', 'entropytime', 'maxAmp', 'sal', 'meanspect', 'skewspect']

    # build a dataset that makes it easy to plot single decoder performance
    g = df.groupby(['bird', 'block', 'segment', 'hemi', 'electrode', 'aprop'])
    for (bird,block,segment,hemi,electrode,aprop),gdf in g:

        assert len(gdf) == 1

        ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == electrode)
        assert ei.sum() == 1
        reg = clean_region(edata.region[ei].values[0])
        dist_l2a = edata.dist_l2a[ei].values[0]
        dist_midline = edata.dist_midline[ei].values[0]

        if bird == 'GreBlu9508M':
            dist_l2a *= 4

        data['bird'].append(bird)
        data['block'].append(block)
        data['segment'].append(segment)
        data['hemi'].append(hemi)
        data['dm'].append(dist_midline)
        data['dl'].append(dist_l2a)
        data['r2'].append(gdf.r2.values[0])
        data['reg'].append(reg)
        data['electrode'].append(electrode)
        data['aprop'].append(aprop)
       
    df = pd.DataFrame(data)
    i = ~np.isnan(df.dm) & ~np.isnan(df.dl) & ~np.isnan(df.r2) & (df.r2 > 0)
    df = df[i]
    print df.describe()

    def _plot_map(_pdata, _ax, _cmap, _maxval, _bgcolor=None, _perf_alpha=False, _plot_region=False, _msize=60):
        if _bgcolor is not None:
            _ax.set_axis_bgcolor(_bgcolor)
        _pval = _pdata['df'].r2.values
        _x = _pdata['df'].dm.values
        _y = _pdata['df'].dl.values
        _regs = _pdata['df'].reg.values

        plt.sca(_ax)
        _alpha = np.ones([len(_pval)])
        if _perf_alpha:
            _alpha = _pdata['df'].r2.values
            _alpha /= _alpha.max()
            _alpha[_alpha > 0.9] = 1.
            _clrs = _cmap(_pval / _maxval)
        else:
            _clrs = _cmap(_pval / _maxval)

        plt.scatter(_x, _y, c=_pval, marker='o', cmap=_cmap, vmin=0, s=_msize, alpha=0.8)
        plt.xlabel('Dist to Midline (mm)')
        plt.ylabel('Dist to L2A (mm)')
        plt.colorbar(label='Decoder R2')
        plt.xlim(0, 2.5)
        plt.ylim(-1, 1)
        """
        for k,(_xx,_yy) in enumerate(zip(_x, _y)):
            plt.plot(_xx, _yy, 'o', c=_clrs[k], alpha=_alpha[k], markersize=_msize)
            if _plot_region:
                plt.text(_xx, _yy, _regs[k], fontsize=8, color='w', alpha=0.7)
        """

    def rb_cmap(x):
        assert np.abs(x).max() <= 1
        _rgb = np.zeros([len(x), 3])
        _pos = x >= 0
        _neg = x < 0

        _rgb[_pos, 0] = x[_pos]
        _rgb[_neg, 2] = np.abs(x[_neg])

        return _rgb

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.05, right=0.95, hspace=0.25, wspace=0.25)
    nrows = 2
    ncols = 3
    for k, aprop in enumerate(aprops_to_show):
        ax = plt.subplot(nrows, ncols, k+1)
        i = df.aprop == aprop
        max_r2 = df[i].r2.max()
        print 'max_r2=%0.2f' % max_r2
        _plot_map({'df':df[i]}, ax, magma, max_r2, _bgcolor='k', _perf_alpha=False, _plot_region=False)
        plt.title(ACOUSTIC_PROP_NAMES[aprop])

    fname = os.path.join(get_this_dir(), 'single_electrode_decoder_r2.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'single_electrode_decoder.h5')
    agg = SingleElectrodeDecoderAggregator.load(agg_file)

    plot_maps(agg)
    # plot_raw_dists()


if __name__ == '__main__':
    set_font()
    draw_figures()
