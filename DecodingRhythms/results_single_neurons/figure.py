import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GMM

from lasp.colormaps import magma
from scipy.stats import chi2

from DecodingRhythms.utils import set_font, COLOR_BLUE_LFP, COLOR_YELLOW_SPIKE
from lasp.plots import multi_plot, custom_legend, grouped_boxplot
from utils import get_this_dir
from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder
from zeebeez.utils import CALL_TYPE_SHORT_NAMES, DECODER_CALL_TYPES


def draw_single_cell(agg, df_cell):

    aprops_to_display = ['category', 'maxAmp', 'sal', 'meanspect', 'q1', 'q2', 'q3',
                         'entropyspect', 'meantime', 'entropytime']

    plist = list()
    for aprop in aprops_to_display:
        perfs = df_cell['perf_%s' % aprop].values
        lkrats = df_cell['lkrat_%s' % aprop].values
        plist.append({'perfs':perfs, 'lkrats':lkrats, 'aprop':aprop})

    def _plot_perf_hist(pdata, ax):
        plt.sca(ax)
        plt.hist(pdata['perfs'], bins=10, color=COLOR_YELLOW_SPIKE)
        plt.xlabel('R2')
        plt.axis('tight')
        plt.xlim(0, 0.4)
        tks = [0.0, 0.1, 0.2, 0.3, 0.4]
        plt.xticks(tks, ['%0.1f' % x for x in tks])
        plt.title(pdata['aprop'])

    def _plot_lkrat_hist(pdata, ax):
        lkr = pdata['lkrats']
        lkr = lkr[lkr >= 0]
        plt.sca(ax)
        plt.hist(lkr, bins=10, color=COLOR_YELLOW_SPIKE)
        plt.xlabel('Likelihood Ratio')
        plt.axis('tight')
        plt.xlim(0, 60)
        plt.title(pdata['aprop'])

    multi_plot(plist, _plot_perf_hist, nrows=2, ncols=5)
    multi_plot(plist, _plot_lkrat_hist, nrows=2, ncols=5)


def draw_perf_corr_mat(agg, df_cell, fmt_str='perf_%s'):

    aprops_to_display = ['category', 'maxAmp', 'sal',
                         'meanspect', 'q1', 'q2', 'q3', 'entropyspect',
                         'meantime', 'kurtosistime', 'entropytime']

    X = np.zeros([len(df_cell), len(aprops_to_display)])
    for k,aprop in enumerate(aprops_to_display):
        X[:, k] = df_cell[fmt_str % aprop].values

    # compute correlation matrix
    C = np.corrcoef(X.T)

    # do PCA
    pca = PCA(n_components=2)
    pca.fit(X)

    pcomps = pca.components_
    ncomps,nf = pcomps.shape
    evar = pca.explained_variance_ratio_

    Xproj = pca.transform(X)

    plt.figure()
    gs = plt.GridSpec(100, 2)

    ax = plt.subplot(gs[:, 0])
    plt.plot(Xproj[:, 0], Xproj[:, 1], 'kx')
    plt.xlabel('PC1')
    plt.xlabel('PC2')

    pheight = 30
    pspace = 10
    for k in range(ncomps):

        si = pspace + k*(pheight+pspace)
        ei = si + pheight

        ax = plt.subplot(gs[si:ei, 1])
        bx = np.arange(nf)
        plt.bar(left=bx, height=np.abs(pcomps[k, :]))
        plt.title('PC %d, evar=%0.2f' % (k+1, evar[k]))
        plt.axis('tight')
        plt.ylim(0, 1)
        if k == ncomps - 1:
            plt.xticks(bx, aprops_to_display, rotation=45)
        else:
            plt.xticks([])
    plt.suptitle('PCA')

    # do clustering
    gmm = GMM(n_components=2)
    gmm.fit(X)
    grps = gmm.predict(X)

    i1 = grps == 0
    i2 = grps == 1

    plt.figure()
    gs = plt.GridSpec(100, 2)

    ax = plt.subplot(gs[:, 0])
    plt.plot(Xproj[i1, 0], Xproj[i1, 1], 'ro', markersize=8., alpha=0.8)
    plt.plot(Xproj[i2, 0], Xproj[i2, 1], 'bo', markersize=8., alpha=0.8)
    plt.xlabel('PC1')
    plt.xlabel('PC2')

    pheight = 30
    pspace = 10
    for k in range(ncomps):
        si = pspace + k*(pheight+pspace)
        ei = si + pheight

        ax = plt.subplot(gs[si:ei, 1])
        bx = np.arange(nf)
        plt.bar(left=bx, height=gmm.means_[k, :])
        plt.title('Mean %d' % (k+1))
        plt.axis('tight')
        plt.ylim(0, 1)
        if k == ncomps - 1:
            plt.xticks(bx, aprops_to_display, rotation=45)
        else:
            plt.xticks([])
    plt.suptitle('Clustering')

    figsize = (18, 12)
    plt.figure(figsize=figsize)

    plt.imshow(C, interpolation='nearest', aspect='auto', cmap=magma, origin='lower', vmin=0, vmax=1)

    tloc = np.arange(len(aprops_to_display))
    plt.xticks(tloc-0.2, aprops_to_display, rotation=45)
    plt.yticks(tloc, aprops_to_display)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Single Neuron Decoder Performance Relationships')


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])
    print '# of groups: %d' % len(g)

    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))
    df_se = pd.read_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_perfs.csv'))
    df_cell = pd.read_csv(os.path.join(data_dir, 'aggregate', 'cell_perfs.csv'))

    # draw_single_cell(agg, df_cell)

    # df0 = df_me[df_me.band == 0]
    # draw_perf_corr_mat(agg, df0, fmt_str='perf_%s_spike')
    # plt.title('Multi-electrode Decoders')

    # draw_perf_corr_mat(agg, df_se)
    # plt.title('Single Electrode LFP Decoders')

    draw_perf_corr_mat(agg, df_cell)

    plt.show()

if __name__ == '__main__':
    set_font()
    draw_figures()

