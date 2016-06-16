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


def export_ds(agg, data_dir='/auto/tdrive/mschachter/data'):

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(),
            'aprop':list(), 'region':list(), 'dist_midline':list(), 'dist_l2a':list(), 'r2':list()}

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))
    i = edata.bird != 'BlaBro09xxF'
    edata = edata[i]

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi', 'electrode', 'aprop'])
    for (bird,block,segment,hemi,electrode,aprop),gdf in g:

        assert len(gdf) == 1

        ei = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == electrode)
        assert ei.sum() == 1
        reg = clean_region(edata.region[ei].values[0])
        dist_l2a = edata.dist_l2a[ei].values[0]
        dist_midline = edata.dist_midline[ei].values[0]

        data['bird'].append(bird)
        data['block'].append(block)
        data['segment'].append(segment)
        data['hemi'].append(hemi)
        data['electrode'].append(electrode)
        data['aprop'].append(aprop)
        data['region'].append(reg)
        data['dist_midline'].append(dist_midline)
        data['dist_l2a'].append(dist_l2a)
        data['r2'].append(gdf.r2.values[0])

    df = pd.DataFrame(data)
    i = ~np.isnan(df.dist_l2a) & ~np.isnan(df.dist_midline)

    df.to_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_decoder.csv'), header=True, index=False)

    return df[i]


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
    aprops_to_show = ['stdtime', 'maxAmp', 'meanspect', 'cvfund', 'sal']

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

        plt.scatter(_x, _y, c=_pval, marker='o', cmap=_cmap, vmin=0, s=_msize, alpha=0.6)
        plt.xlabel('Dist to Midline (mm)')
        plt.ylabel('Dist to L2A (mm)')
        cbar = plt.colorbar(label='Decoder R2')
        _new_ytks = ['%0.2f' % float(_yt.get_text()) for _yt in cbar.ax.get_yticklabels()]
        print '_new_ytks=',_new_ytks
        cbar.ax.set_yticklabels(_new_ytks)
        # print 'ytks=',_ytks
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
    fig.subplots_adjust(left=0.05, right=0.98, hspace=0.25, wspace=0.25)
    nrows = 2
    ncols = 3
    for k, aprop in enumerate(aprops_to_show):
        ax = plt.subplot(nrows, ncols, k+1)
        i = df.aprop == aprop
        max_r2 = df[i].r2.max()
        print 'max_r2=%0.2f' % max_r2
        # _plot_map({'df':df[i]}, ax, magma, max_r2, _bgcolor='k', _perf_alpha=False, _plot_region=False)
        _plot_map({'df': df[i]}, ax, plt.cm.afmhot_r, max_r2, _bgcolor='w', _perf_alpha=False, _plot_region=False)
        plt.title(ACOUSTIC_PROP_NAMES[aprop])

    ax = plt.subplot(nrows, ncols, 6)
    plot_r2_region_prop(ax)

    fname = os.path.join(get_this_dir(), 'single_electrode_decoder_r2.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def plot_cov_mat(agg):

    # plot a covariance matrix of r2s
    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi', 'electrode'])
    r2_by_aprop = dict()
    for aprop in ALL_ACOUSTIC_PROPS:
        r2_by_aprop[aprop] = list()
    for (bird, block, segment, hemi, electrode), gdf in g:

        assert len(gdf) == len(ALL_ACOUSTIC_PROPS)
        for aprop in ALL_ACOUSTIC_PROPS:
            i = gdf.aprop == aprop
            assert i.sum() == 1
            r2_by_aprop[aprop].append(gdf.r2[i].values[0])

    X = np.zeros([len(ALL_ACOUSTIC_PROPS), len(r2_by_aprop['sal'])])
    for k,aprop in enumerate(ALL_ACOUSTIC_PROPS):
        X[k, :] = r2_by_aprop[aprop]

    C = np.corrcoef(X)

    fig = plt.figure()
    fig.subplots_adjust(top=0.98, bottom=0.15)
    plt.imshow(C, interpolation='nearest', aspect='auto', vmin=-1, vmax=1, cmap=plt.cm.seismic, origin='lower')
    nprops = len(ALL_ACOUSTIC_PROPS)
    plt.xticks(range(nprops), ALL_ACOUSTIC_PROPS, rotation=90)
    plt.yticks(range(nprops), ALL_ACOUSTIC_PROPS)
    plt.colorbar()
    plt.show()


def plot_r2_region_prop(ax=None, data_dir='/auto/tdrive/mschachter/data'):

    aprops_to_show = ['stdtime', 'maxAmp', 'meanspect', 'cvfund', 'sal']
    regs = ['CM', 'L1', 'L2', 'L3', 'NCM']

    df = pd.read_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_decoder.csv'))

    r2_mat = np.zeros([len(aprops_to_show), len(regs)])
    for k,aprop in enumerate(aprops_to_show):
        for j,reg in enumerate(regs):
            i = (df.aprop == aprop) & (df.region == reg)
            r2_mat[k, j] = df.r2[i].mean()

    if ax is None:
        plt.figure()
        ax = plt.gca()
    plt.sca(ax)

    ax.set_axis_bgcolor('w')
    plt.imshow(r2_mat, interpolation='nearest', aspect='auto', vmin=0, cmap=plt.cm.afmhot_r, origin='upper')
    plt.xticks(range(len(regs)), regs)
    plt.yticks(range(len(aprops_to_show)), [ACOUSTIC_PROP_NAMES[aprop] for aprop in aprops_to_show])

    plt.colorbar(label='Decoder R2')


def draw_figures(data_dir='/auto/tdrive/mschachter/data', fig_dir='/auto/tdrive/mschachter/figures/encoder+decoder'):

    agg_file = os.path.join(data_dir, 'aggregate', 'single_electrode_decoder.h5')
    agg = SingleElectrodeDecoderAggregator.load(agg_file)

    plot_maps(agg)
    # plot_raw_dists()
    # plot_cov_mat(agg)
    # plot_r2_region_prop()
    # export_ds(agg)
    plt.show()


if __name__ == '__main__':
    set_font()
    draw_figures()
