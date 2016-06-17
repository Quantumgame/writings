import os
import operator

import numpy as np
import matplotlib.pyplot as plt

from lasp.plots import multi_plot
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    bird = 'GreBlu9508M'
    block = 'Site4'
    seg = 'Call1'
    hemi = 'L'
    file_ext = '_'.join([bird, block, seg, hemi])

    pcf_file = os.path.join(data_dir, bird, 'transforms', 'PairwiseCF_%s_%s_new.h5' % (file_ext, 'raw'))
    pcf = PairwiseCFTransform.load(pcf_file)

    g = pcf.df.groupby(['stim_id', 'order', 'stim_type'])

    plist = list()

    i = (pcf.df.decomp == 'full') & (pcf.df.electrode1 == pcf.df.electrode2)
    assert i.sum() > 0
    xfull_indices = list(pcf.df['index'][i].values)
    print 'len(pcf.df)=%d' % len(pcf.df)
    print 'pcf.df[index].max()=%d' % pcf.df['index'].max()
    print 'pcf.psds.shape=',pcf.psds.shape
    print 'xfull_indices.max()=%d' % max(xfull_indices)
    print 'len(xfull_indices)=%d' % len(xfull_indices)
    Xfull = pcf.psds[xfull_indices, :]
    Xfull /= Xfull.max()
    pcf.log_transform(Xfull)
    Xfull -= Xfull.mean(axis=0)
    Xfull /= Xfull.std(axis=0, ddof=1)

    # take log transform of power spectra
    i = (pcf.df.decomp == 'onewin') & (pcf.df.electrode1 == pcf.df.electrode2)
    assert i.sum() > 0
    xone_indices = list(pcf.df['index'][i].values)
    Xonewin = pcf.psds[xone_indices, :]
    Xonewin /= Xonewin.max()
    pcf.log_transform(Xonewin)
    Xonewin -= Xonewin.mean(axis=0)
    Xonewin /= Xonewin.std(axis=0, ddof=1)

    for (stim_id,order,stim_type),gdf in g:

        electrodes = gdf.electrode1.unique()
        stim_dur = gdf.stim_duration.values[0]
        if stim_dur < 0.050 or stim_dur > 0.400:
            continue

        for e in electrodes:
            i = (gdf.decomp == 'full') & (gdf.electrode1 == e) & (gdf.electrode1 == gdf.electrode2)
            assert i.sum() == 1

            xi = gdf['index'][i].values[0]
            ii = xfull_indices.index(xi)
            full_psd = Xfull[ii, :]

            i = (gdf.decomp == 'onewin') & (gdf.electrode1 == e) & (gdf.electrode1 == gdf.electrode2)
            assert i.sum() == 1
            xi = gdf['index'][i].values[0]
            ii = xone_indices.index(xi)
            onewin_psd = Xonewin[ii, :]

            plist.append({'full_psd':full_psd, 'onewin_psd':onewin_psd, 'stim_id':stim_id, 'stim_order':order,
                          'stim_type':stim_type, 'electrode':e, 'stim_dur':stim_dur})

    np.random.shuffle(plist)
    plist.sort(key=operator.itemgetter('stim_dur'))
    short_plist = [x for k,x in enumerate(plist) if k % 20 == 0]
    print 'len(short_plist)=%d' % len(short_plist)

    def _plot_psds(_pdata, _ax):
        absmax = max(np.abs(_pdata['full_psd']).max(), np.abs(_pdata['onewin_psd']).max())
        plt.axhline(0, c='k')
        plt.plot(pcf.freqs, _pdata['full_psd'], 'k-', linewidth=3.0, alpha=0.7)
        plt.plot(pcf.freqs, _pdata['onewin_psd'], 'g-', linewidth=3.0, alpha=0.7)
        plt.title('e%d: %d_%d (%s) %0.3fs' % (_pdata['electrode'], _pdata['stim_id'], _pdata['stim_order'], _pdata['stim_type'], _pdata['stim_dur']))
        plt.axis('tight')
        plt.ylim(-absmax, absmax)

    multi_plot(short_plist, _plot_psds, nrows=5, ncols=9)
    plt.show()


if __name__ == '__main__':

    draw_figures()