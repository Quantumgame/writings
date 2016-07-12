import os

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from lasp.plots import custom_legend, multi_plot
from zeebeez.aggregators.biosound import AggregateBiosounds
from zeebeez.utils import DECODER_CALL_TYPES, CALL_TYPE_COLORS


def draw_joint_relationships(data_dir='/auto/tdrive/mschachter/data'):

    big3_file = os.path.join(data_dir, 'aggregate', 'acoustic_big3.csv')

    if not os.path.exists(big3_file):
        bs_file = os.path.join(data_dir, 'aggregate', 'biosound.h5')
        bs_agg = AggregateBiosounds.load(bs_file)

        aprops = ['maxAmp', 'sal', 'meanspect']
        Xz, good_indices = bs_agg.remove_duplicates(aprops=aprops, thresh=np.inf)
        stim_types = bs_agg.df.stim_type[good_indices].values

        pdata = {'maxAmp': Xz[:, 0], 'sal': Xz[:, 1], 'meanspect': Xz[:, 2], 'stim_type': stim_types}
        df = pd.DataFrame(pdata)
        df.to_csv(big3_file, header=True, index=False)
    else:
        df = pd.read_csv(big3_file)

    print '# of syllables used: %d' % len(df)

    plt.figure()
    ax = plt.subplot(111, projection='3d')

    maxAmp_rescaled = df.maxAmp
    maxAmp_rescaled -= maxAmp_rescaled.min()
    maxAmp_rescaled = np.log(maxAmp_rescaled + 1e-6)
    maxAmp_rescaled -= maxAmp_rescaled.mean()
    maxAmp_rescaled /= maxAmp_rescaled.std(ddof=1)

    for ct in DECODER_CALL_TYPES:
        i = df.stim_type == ct
        if i.sum() == 0:
            continue

        x = maxAmp_rescaled[i]
        y = df.sal[i].values
        z = df.meanspect[i].values
        c = CALL_TYPE_COLORS[ct]
        ax.scatter(x, y, z, s=49, c=c)

    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Saliency')
    ax.set_zlabel('Mean Frequency')

    leg = custom_legend([CALL_TYPE_COLORS[ct] for ct in DECODER_CALL_TYPES], DECODER_CALL_TYPES)
    plt.legend(handles=leg)

    plt.figure()
    nrows = 1
    ncols = 3

    call_type_order = ['song', 'Te', 'DC', 'Be', 'LT', 'Ne', 'Ag', 'Di', 'Th']

    # plot amplitude vs saliency
    ax = plt.subplot(nrows, ncols, 1)
    for ct in call_type_order:
        i = df.stim_type == ct
        if i.sum() == 0:
            continue
        x = maxAmp_rescaled[i]
        y = df.sal[i].values
        c = CALL_TYPE_COLORS[ct]
        ax.scatter(x, y, s=49, c=c, alpha=0.7)
    plt.xlabel('Maximum Amplitude')
    plt.ylabel('Saliency')
    plt.xlim(-3, 2)

    # plot amplitude vs meanspect
    ax = plt.subplot(nrows, ncols, 2)
    for ct in call_type_order:
        i = df.stim_type == ct
        if i.sum() == 0:
            continue
        x = maxAmp_rescaled[i]
        y = df.meanspect[i].values
        c = CALL_TYPE_COLORS[ct]
        ax.scatter(x, y, s=49, c=c, alpha=0.7)
    plt.xlabel('Maximum Amplitude')
    plt.ylabel('Mean Spectral Freq')
    plt.xlim(-3, 2)

    # plot saliency vs meanspect
    ax = plt.subplot(nrows, ncols, 3)
    for ct in call_type_order:
        i = df.stim_type == ct
        if i.sum() == 0:
            continue
        x = df.sal[i].values
        y = df.meanspect[i].values
        c = CALL_TYPE_COLORS[ct]
        ax.scatter(x, y, s=49, c=c, alpha=0.7)
    plt.xlabel('Saliency')
    plt.ylabel('Mean Spectral Freq')

    plt.show()


def draw_neural_joint_relationships(preproc_file):

    hf = h5py.File(preproc_file)
    index2prop = list(hf.attrs['integer2prop'])
    index2type = list(hf.attrs['integer2type'])
    S = np.array(hf['S'])
    X = np.array(hf['X'])
    Y = np.array(hf['Y'])
    hf.close()

    stim_type = [index2type[k] for k in Y[:, 0]]

    sub_i = [index2prop.index('maxAmp'), index2prop.index('sal'), index2prop.index('meanspect')]
    S = S[:, sub_i]

    ncells = X.shape[1]

    plist = list()
    for n in range(ncells):
        plist.append({'n':n, 'x':S[:, 0], 'y':S[:, 1], 'r':X[:, n], 'xlabel':'maxAmp', 'ylabel':'saliency', 'logx':True})
        plist.append({'n': n, 'x': S[:, 0], 'y': S[:, 2], 'r': X[:, n], 'xlabel': 'maxAmp', 'ylabel': 'meanspect', 'logx':True})
        plist.append({'n': n, 'x': S[:, 1], 'y': S[:, 2], 'r': X[:, n], 'xlabel': 'saliency', 'ylabel': 'meanspect', 'logx':False})

    def _plot_scatter(_pdata, _ax):
        if _pdata['logx']:
            _ax.set_xscale('log')
        """
        _clrs = [CALL_TYPE_COLORS[ct] for ct in stim_type]
        _r = _pdata['r']
        _r -= _r.min()
        _r /= _r.max()
        for _x,_y,_r,_c in zip(_pdata['x'], _pdata['y'], _r, _clrs):
            plt.plot(_x, _y, 'o', c=_c, alpha=_r)
        """
        plt.scatter(_pdata['x'], _pdata['y'], marker='o', c=_pdata['r'], cmap=plt.cm.afmhot_r, s=49, alpha=0.7)

        plt.xlabel(_pdata['xlabel'])
        plt.ylabel(_pdata['ylabel'])

        plt.title('cell %d' % _pdata['n'])
        plt.colorbar(label='Spike Rate')

    multi_plot(plist, _plot_scatter, nrows=3, ncols=3, figsize=(23,13), wspace=0.25, hspace=0.25)
    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    # draw_joint_relationships()

    preproc_file = os.path.join(data_dir, 'GreBlu9508M', 'preprocess', 'preproc_Site4_Call1_L_spike_rate.h5')
    draw_neural_joint_relationships(preproc_file)

if __name__ == '__main__':
    draw_figures()


