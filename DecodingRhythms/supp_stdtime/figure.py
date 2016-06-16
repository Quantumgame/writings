import os
import operator

import numpy as np
import matplotlib.pyplot as plt

from lasp.signal import bandpass_filter
from lasp.sound import plot_spectrogram, spec_colormap
from DecodingRhythms.utils import get_full_data
from zeebeez.aggregators.tuning_curve import TuningCurveAggregator


def draw_tuning_curves(data_dir='/auto/tdrive/mschachter/data'):
    agg_file = os.path.join(data_dir, 'aggregate', 'tuning_curve.h5')
    agg = TuningCurveAggregator.load(agg_file)

    r2_thresh = 0.01
    i = (agg.df.decomp == 'full_psds') & (agg.df.freq == 99) & (agg.df.aprop == 'stdtime') & (agg.df.r2 > r2_thresh)
    assert i.sum() > 0

    xindex = agg.df.xindex[i]
    assert len(xindex) > 0

    lst = zip(xindex, agg.df.r2[i].values)
    lst.sort(key=operator.itemgetter(1))
    xindex = [x[0] for x in lst]

    cx = agg.curve_x[xindex, :]
    tc = agg.tuning_curves[xindex, :]

    plt.figure()
    for x, y in zip(cx, tc):
        plt.plot(x, y, 'k-', alpha=0.7)
    plt.xlabel('stdtime')
    plt.ylabel('Power (99Hz)')
    plt.xlim(0.02, 0.09)

    plt.show()


def draw_raw_lfp():

    spec_colormap()
    d = get_full_data('GreBlu9508M', 'Site4', 'Call1', 'L', 284)

    ntrials = 5
    trial_indices = range(10)
    np.random.shuffle(trial_indices)

    the_lfp = d['lfp'][trial_indices[:ntrials], :, :]

    ntrials2, nelectrodes,nt = the_lfp.shape
    bp_lfp = np.zeros_like(the_lfp)
    # bandpass from 95-105Hz
    for k in range(ntrials):
        for n in range(nelectrodes):
            bp_lfp[k, n, :] = bandpass_filter(the_lfp[k, n, :], d['lfp_sample_rate'], 30., 80.)

    bp_lfp = bp_lfp**2
    lfp_t = np.arange(nt) / d['lfp_sample_rate']

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)

    gs = plt.GridSpec(2 + ntrials, 1)

    ax = plt.subplot(gs[0, 0])
    plot_spectrogram(d['spec_t'], d['spec_freq'], d['spec'], ax=ax, colormap='SpectroColorMap', colorbar=True)

    the_lfp = the_lfp.mean(axis=0)
    ax = plt.subplot(gs[1, 0])
    absmax = np.abs(the_lfp).max()
    plt.imshow(the_lfp, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-absmax, vmax=absmax,
               extent=(lfp_t.min(), lfp_t.max(), 0, nelectrodes))
    plt.colorbar()

    for k in range(ntrials):
        ax = plt.subplot(gs[2+k, 0])
        absmax = np.abs(bp_lfp[k, :, :]).max()
        plt.imshow(bp_lfp[k, :, :], interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot_r, vmin=0, vmax=absmax,
                   extent=(lfp_t.min(), lfp_t.max(), 0, nelectrodes))
        plt.colorbar()

    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    # draw_tuning_curves()
    draw_raw_lfp()


if __name__ == '__main__':
    draw_figures()
