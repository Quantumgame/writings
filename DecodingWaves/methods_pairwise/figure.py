import os
import operator

import numpy as np
import matplotlib.pyplot as plt
from lasp.plots import custom_legend
from lasp.signal import correlation_function, coherency, break_envelope_into_events
from lasp.sound import plot_spectrogram, spec_colormap
from utils import get_this_dir, set_font, compute_spectra_and_coherence_multi_electrode_single_trial, get_freqs, \
    get_lags_ms, get_psd_stats

from zeebeez.core.experiment import Experiment
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def draw_figures(bird, block, segment, hemi, e1, e2, stim_id, trial, syllable_index, data_dir='/auto/tdrive/mschachter/data', exp=None):

    # load up the experiment
    if exp is None:
        bird_dir = os.path.join(data_dir, bird)
        exp_file = os.path.join(bird_dir, '%s.h5' % bird)
        stim_file = os.path.join(bird_dir, 'stims.h5')
        exp = Experiment.load(exp_file, stim_file)
    seg = exp.get_segment(block, segment)

    # get the start and end times of the stimulus
    etable = exp.get_epoch_table(seg)

    i = etable['id'] == stim_id
    stim_times = zip(etable[i]['start_time'].values, etable[i]['end_time'].values)
    stim_times.sort(key=operator.itemgetter(0))

    start_time,end_time = stim_times[trial]
    stim_dur = float(end_time - start_time)

    # get a slice of the LFP
    lfp_data = exp.get_lfp_slice(seg, start_time, end_time, zscore=True)
    electrode_indices,lfps,sample_rate = lfp_data[hemi]

    # get the log spectrogram of the stimulus
    stim_spec_t,stim_spec_freq,stim_spec = exp.get_spectrogram_slice(seg, start_time, end_time)
    stim_spec_t = np.linspace(0, stim_dur, len(stim_spec_t))
    stim_spec_dt = np.diff(stim_spec_t)[0]
    nz = stim_spec > 0
    stim_spec[nz] = 20*np.log10(stim_spec[nz]) + 100
    stim_spec[stim_spec < 0] = 0

    # get the amplitude envelope
    amp_env = stim_spec.std(axis=0, ddof=1)
    amp_env -= amp_env.min()
    amp_env /= amp_env.max()

    # segment the amplitude envelope into syllables
    merge_thresh = int(0.002*sample_rate)
    events = break_envelope_into_events(amp_env, threshold=0.05, merge_thresh=merge_thresh)

    # translate the event indices into actual times
    events *= stim_spec_dt

    syllable_start,syllable_end,syllable_max_amp = events[syllable_index]
    amp_env_rs = amp_env*(stim_spec_freq.max() - stim_spec_freq.min()) + stim_spec_freq.min()
    last_syllable_end = events[-1, 1] + 0.025

    i1 = electrode_indices.index(e1)
    i2 = electrode_indices.index(e2)

    lfp1 = lfps[i1, :]
    lfp2 = lfps[i2, :]

    t = np.arange(len(lfp1)) / sample_rate
    legend = ['E%d' % e1, 'E%d' % e2]

    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    # get the power spectrum stats for this site
    psd_stats = get_psd_stats(bird, block, segment, hemi)

    # compute the power spectra and cross coherence for all electrodes
    lags_ms = get_lags_ms(sample_rate)
    spectra,cross_mat = compute_spectra_and_coherence_multi_electrode_single_trial(lfps, sample_rate, electrode_indices, electrode_order, psd_stats=psd_stats)

    # plot the stimulus and raw LFP for two electrodes
    figsize = (24.0, 10)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

    ax = plt.subplot(2, 1, 1)
    plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, ticks=True, fmin=300, fmax=8000,
                     colormap='SpectroColorMap', colorbar=False)
    # plt.plot(stim_spec_t, amp_env_rs, 'k-', linewidth=2.0, alpha=0.50)
    plt.axvline(syllable_start, c='k', linestyle='dashed', linewidth=3.0)
    plt.axvline(syllable_end, c='k', linestyle='dashed', linewidth=3.0)
    plt.axis('tight')
    plt.xlim(0, last_syllable_end)

    ax = plt.subplot(2, 1, 2)
    plt.plot(t, lfp1, 'b-', linewidth=3.0)
    plt.plot(t, lfp2, 'r-', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_start, c='k', linestyle='dashed', linewidth=3.0)
    plt.axvline(syllable_end, c='k', linestyle='dashed', linewidth=3.0)
    plt.xlabel('Time (ms)')
    plt.ylabel('LFP (z-scored)')
    plt.legend(legend)
    plt.axis('tight')
    plt.xlim(0, last_syllable_end)

    fname = os.path.join(get_this_dir(), 'raw.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    # restrict the lfps to a single syllable
    syllable_si = int(syllable_start*sample_rate)
    syllable_ei = int(syllable_end*sample_rate)
    lfp1 = lfp1[syllable_si:syllable_ei]
    lfp2 = lfp2[syllable_si:syllable_ei]

    # plot the two power spectra
    psd_ub = 6
    psd_lb = 0
    i1 = electrode_order.index(e1)
    i2 = electrode_order.index(e2)

    a1 = spectra[i1, :]
    a2 = spectra[i2, :]
    freqs = get_freqs(sample_rate)

    figsize = (10.0, 4.0)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.90, bottom=0.10, left=0.10, right=0.99, hspace=0.10)

    ax = plt.subplot(1, 2, 1)
    plt.plot(freqs, a1, 'b-', linewidth=3.0)
    plt.plot(freqs, a2, 'r-', linewidth=3.0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (z-scored)')
    plt.title('Power Spectrum')
    handles = custom_legend(['b', 'r'], legend)
    plt.legend(handles=handles, fontsize='small')
    plt.axis('tight')
    plt.ylim(psd_lb, psd_ub)

    # plot the coherency
    cf_lb = -0.1
    cf_ub = 0.3
    coh = cross_mat[i1, i2, :]
    ax = plt.subplot(1, 2, 2)
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.plot(lags_ms, coh, 'g-', linewidth=3.0)
    plt.xlabel('Frequency (Hz)')
    plt.title('Coherency')
    plt.axis('tight')
    plt.ylim(cf_lb, cf_ub)

    fname = os.path.join(get_this_dir(), 'auto+cross.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    # compute all cross and auto-correlations
    nelectrodes = len(electrode_order)

    # make a plot
    figsize = (24.0, 13.5)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)

    gs = plt.GridSpec(nelectrodes, nelectrodes)
    for i in range(nelectrodes):
        for j in range(i+1):
            ax = plt.subplot(gs[i, j])
            plt.axhline(0, c='k')
            plt.axvline(0, c='k')

            _e1 = electrode_order[i]
            _e2 = electrode_order[j]

            clr = 'k'
            if i == j:
                if _e1 == e1:
                    clr = 'b'
                elif _e1 == e2:
                    clr = 'r'
            else:
                if _e1 == e1 and _e2 == e2:
                    clr = 'g'

            if i == j:
                plt.plot(freqs, spectra[i, :], '-', c=clr, linewidth=2.0)
            else:
                plt.plot(lags_ms, cross_mat[i, j, :], '-', c=clr, linewidth=2.0)
            plt.xticks([])
            plt.yticks([])
            plt.axis('tight')
            if i != j:
                plt.ylim(cf_lb, cf_ub)
            else:
                plt.axhline(0, c='k')
                plt.ylim(psd_lb, psd_ub)

            if j == 0:
                plt.ylabel('E%d' % electrode_order[i])
            if i == nelectrodes-1:
                plt.xlabel('E%d' % electrode_order[j])

    fname = os.path.join(get_this_dir(), 'cross_all.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':

    spec_colormap()
    set_font()

    bird = 'GreBlu9508M'
    block = 'Site4'
    segment = 'Call1'
    hemi = 'L'
    e1 = 6
    e2 = 4
    stim_id = 43
    trial = 3
    syllable_index = 4

    draw_figures(bird, block, segment, hemi, e1, e2, stim_id, trial, syllable_index)
    plt.show()
