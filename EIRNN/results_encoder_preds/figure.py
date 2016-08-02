import os
import operator

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import get_this_dir, set_font
from lasp.sound import plot_spectrogram, spec_colormap
from zeebeez.transforms.rnn_preprocess import RNNPreprocessTransform
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    bird = 'GreBlu9508M'
    block = 'Site4'
    segment = 'Call1'
    hemi = 'L'
    fname = '%s_%s_%s_%s' % (bird, block, segment, hemi)
    exp_dir = os.path.join(data_dir, bird)

    preproc_file = os.path.join(exp_dir, 'preprocess', 'RNNPreprocess_%s.h5' % fname)
    rp = RNNPreprocessTransform.load(preproc_file)

    pred_file = os.path.join(exp_dir, 'rnn', 'LFPEnvelope_%s.h5' % fname)
    hf = h5py.File(pred_file, 'r')
    Ypred = np.array(hf['Ypred'])
    hf.close()

    assert Ypred.shape[0] == rp.U.shape[0]

    stim_id = 277
    trial = 5

    i = (rp.event_df.stim_id == stim_id) & (rp.event_df.trial == trial)
    assert i.sum() == 1
    start_time = rp.event_df[i].start_time.values[0]
    end_time = rp.event_df[i].end_time.values[0]
    si = int(start_time*rp.sample_rate)
    ei = int(end_time * rp.sample_rate)

    spec = rp.U[si:ei, :].T
    spec_freq = rp.spec_freq
    spec_env = spec.sum(axis=0)
    spec_env /= spec_env.max()

    lfp = rp.Yraw[si:ei, :].T
    lfp_pred = Ypred[si:ei, :].T

    nt = spec.shape[1]
    t = np.arange(nt) / rp.sample_rate

    index2electrode = list(rp.index2electrode)
    electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    if hemi == 'R':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    fig = plt.figure(figsize=(23, 13), facecolor='w')
    gs = plt.GridSpec(100, 1)

    ax = plt.subplot(gs[:25, :])
    spec_env *= (spec_freq.max() - spec_freq.min())
    spec_env += spec_freq.min()
    plot_spectrogram(t, spec_freq, spec, ax=ax, ticks=True, fmax=8000., colormap='SpectroColorMap', colorbar=False)
    plt.plot(t, spec_env, 'k-', linewidth=5.0, alpha=0.7)
    plt.axis('tight')

    ax = plt.subplot(gs[30:, :])
    nelectrodes = len(index2electrode)
    lfp_spacing = 5.
    for k in range(nelectrodes):
        e = electrode_order[nelectrodes-k-1]
        n = index2electrode.index(e)
        offset = k*lfp_spacing
        plt.plot(t, lfp[n, :] + offset, 'k-', alpha=0.7, linewidth=5.0)
        plt.plot(t, lfp_pred[n, :] + offset, 'r-', alpha=0.7, linewidth=5.0)
    plt.yticks([])
    plt.axis('tight')
    plt.xlabel('Time (s)')
    plt.ylabel('LFP')

    fname = os.path.join(get_this_dir(), 'encoder_pred.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()


if __name__ == '__main__':
    spec_colormap()
    set_font()
    draw_figures()

