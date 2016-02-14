import os

import h5py

import numpy as np
import matplotlib.pyplot as plt
from lasp.plots import multi_plot

from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT


def draw_figure(data_dir='/auto/tdrive/mschachter/data', bird='GreBlu9508M'):

    pp_file = os.path.join(data_dir, bird, 'preprocess', 'preproc_raw_coherence_band0_npc0_self_locked_all_Site4_Call1_L.h5')
    print 'pp_file=%s' % pp_file

    hf = h5py.File(pp_file, 'r')
    X = np.array(hf['X'])
    S = np.array(hf['S'])
    index2electrode = list(hf.attrs['index2electrode'])
    index2aprop = list(hf.attrs['integer2prop'])
    freqs = list(hf.attrs['freqs'])
    hf.close()

    reduced_aprops = ['fund',
                       'sal',
                       'voice2percent',
                       'meanspect',
                       'skewspect',
                       'entropyspect',
                       'q2',
                       'meantime',
                       'skewtime',
                       'entropytime',
                       'maxAmp']

    nelectrodes = len(index2electrode)
    nfreqs = len(freqs)

    electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT

    X -= X.mean()
    X /= X.std(ddof=1)
    S -= S.mean()
    S /= S.std(ddof=1)

    cc_mats = list()
    for k,aprop in enumerate(reduced_aprops):
        s = S[:, k]
        CC = np.zeros([nelectrodes, nfreqs])
        for j,e in enumerate(index2electrode):
            for m,f in enumerate(freqs):

                i = j*nfreqs + m
                x = X[:, i]
                jj = electrode_order.index(e)
                CC[jj, m] = np.corrcoef(x, s)[0, 1]
        cc_mats.append({'CC':CC, 'aprop':aprop})

    def _plot_cc_mat(pdata, ax):
        plt.sca(ax)
        plt.imshow(pdata['CC'], interpolation='nearest', aspect='auto', vmin=-0.5, vmax=0.5, cmap=plt.cm.seismic)
        plt.yticks(range(nelectrodes), ['%d' % _e for _e in electrode_order])
        plt.xticks(range(nfreqs), ['%d' % _f for _f in freqs])
        plt.title(pdata['aprop'])
        plt.colorbar()

    multi_plot(cc_mats, _plot_cc_mat, nrows=4, ncols=3)
    plt.show()


if __name__ == '__main__':
    draw_figure()
