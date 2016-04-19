import os
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasp.plots import multi_plot

from utils import set_font
from zeebeez.aggregators.pard import PARDAggregator
from zeebeez.utils import REDUCED_ACOUSTIC_PROPS


def get_freqs_and_lags():
    #TODO hack
    hf = h5py.File('/auto/tdrive/mschachter/data/GreBlu9508M/preprocess/preproc_self_locked_Site4_Call1_L.h5')
    lags = hf.attrs['lags']
    freqs = hf.attrs['freqs']
    hf.close()
    nlags = len(lags)

    return freqs,lags


def draw_all_weights(agg, data_dir='/auto/tdrive/mschachter/data',
                     fig_dir='/auto/tdrive/mschachter/figures/decoder_weights'):

    freqs,lags = get_freqs_and_lags()

    font = {'family':'normal', 'weight':'bold', 'size':10}
    plt.matplotlib.rc('font', **font)

    i = agg.df.decomp == 'self_locked'
    g = agg.df[i].groupby(['bird', 'block', 'segment', 'hemi'])

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))
    
    wdata = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'aprop':list(), 'windex':list(), 'site':list(), 'r2':list()}
    W = list()

    for (bird,block,seg,hemi),gdf in g:

        assert len(gdf) == 1

        wkey = gdf['wkey'].values[0]
        iindex = gdf['iindex'].values[0]
        index2electrode = agg.index2electrode[iindex]

        dperf = agg.decoder_perfs[wkey]
        dweights = agg.decoder_weights[wkey]

        for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
            wdata['bird'].append(bird)
            wdata['block'].append(block)
            wdata['segment'].append(seg)
            wdata['hemi'].append(hemi)
            wdata['aprop'].append(aprop)
            wdata['r2'].append(dperf[k])
            wdata['site'].append('%s_%s_%s_%s' % (bird, block, seg, hemi))
            wdata['windex'].append(len(W))

            W.append(dweights[:, :, k])

    W = np.array(W)
    wdf = pd.DataFrame(wdata)

    def _plot_weights(_pdata, _ax):
        plt.sca(_ax)
        _absmax = np.abs(_pdata['W']).max()
        plt.imshow(_pdata['W'], interpolation='nearest', aspect='auto', vmin=-_absmax, vmax=_absmax, cmap=plt.cm.seismic)
        plt.xticks(range(len(freqs)), ['%d' % x for x in freqs])
        plt.title(_pdata['title'])

    for k,aprop in enumerate(REDUCED_ACOUSTIC_PROPS):
        plist = list()
        i = wdf.aprop == aprop
        sites = wdf[i].site.values
        r2s = wdf[i].r2.values
        windex = wdf[i].windex.values

        for (site,r2,wi) in zip(sites, r2s,windex):
            plist.append({'title':'%s %0.2f' % (site,r2), 'W':W[wi, :, :]})

        multi_plot(plist, _plot_weights, title=aprop, nrows=4, ncols=6, figsize=(23, 13))
        plt.savefig(os.path.join(fig_dir, 'decoder_weights_%s.png' % aprop))
        plt.close('all')

def draw_figures(data_dir='/auto/tdrive/mschachter/data'):
    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = PARDAggregator.load(agg_file)
    draw_all_weights(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()
