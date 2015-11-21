import os
import operator

import numpy as np

import matplotlib.pyplot as plt

from lasp.plots import custom_legend
from lasp.signal import break_envelope_into_events, correlation_function, coherency
from lasp.sound import plot_spectrogram, spec_colormap

from zeebeez.core.experiment import Experiment

from utils import get_this_dir, set_font, compute_spectra_and_coherence_single_electrode, get_lags_ms, \
    compute_avg_and_ms, get_psd_stats


def draw_figures(bird, block, segment, hemi, e1, e2, stim_id, syllable_index, data_dir='/auto/tdrive/mschachter/data', exp=None):

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
    stim_times = np.array(stim_times)

    stim_durs = stim_times[:, 1] - stim_times[:, 0]
    stim_dur = stim_durs.min()

    # aggregate the LFPs across trials
    lfps = list()
    sample_rate = None
    electrode_indices = None
    for start_time,end_time in stim_times:

        # get a slice of the LFP
        lfp_data = exp.get_lfp_slice(seg, start_time, end_time)
        electrode_indices,the_lfps,sample_rate = lfp_data[hemi]

        stim_dur_i = int(stim_dur*sample_rate)
        lfps.append(the_lfps[:, :stim_dur_i])
    lfps = np.array(lfps)

    # rescale the LFPs to they are in uV
    lfps *= 1e6

    # get the log spectrogram of the stimulus
    start_time_0 = stim_times[0][0]
    end_time_0 = stim_times[0][1]
    stim_spec_t,stim_spec_freq,stim_spec = exp.get_spectrogram_slice(seg, start_time_0, end_time_0)
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
    syllable_start -= 0.005
    syllable_end += 0.010
    amp_env_rs = amp_env*(stim_spec_freq.max() - stim_spec_freq.min()) + stim_spec_freq.min()
    last_syllable_end = events[-1, 1] + 0.025

    i1 = electrode_indices.index(e1)
    i2 = electrode_indices.index(e2)

    lfp1 = lfps[:, i1, :]
    lfp2 = lfps[:, i2, :]

    # zscore the LFP
    lfp1 -= lfp1.mean()
    lfp1 /= lfp1.std(ddof=1)
    lfp2 -= lfp2.mean()
    lfp2 /= lfp2.std(ddof=1)

    ntrials,nelectrodes,nt = lfps.shape
    ntrials_to_plot = 5

    t = np.arange(nt) / sample_rate

    # plot the stimulus and raw LFP for two electrodes
    figsize = (24.0, 10)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.99, hspace=0.10)
    gs = plt.GridSpec(3, 100)

    ax = plt.subplot(gs[0, :60])
    plot_spectrogram(stim_spec_t, stim_spec_freq, stim_spec, ax=ax, ticks=True, fmin=300, fmax=8000,
                     colormap='SpectroColorMap', colorbar=False)
    # plt.plot(stim_spec_t, amp_env_rs, 'k-', linewidth=2.0, alpha=0.50)
    plt.axvline(syllable_start, c='k', linestyle='dashed', linewidth=3.0)
    plt.axvline(syllable_end, c='k', linestyle='dashed', linewidth=3.0)
    plt.axis('tight')
    plt.xlim(0, last_syllable_end)

    # plot the first LFP (all trials)
    ax = plt.subplot(gs[1, :60])
    for k in range(ntrials_to_plot):
        plt.plot(t, lfp1[k, :], '-', linewidth=2.0, alpha=0.75)
    plt.plot(t, lfp1.mean(axis=0), 'k-', linewidth=3.0)
    plt.axvline(syllable_start, c='k', linestyle='dashed', linewidth=3.0)
    plt.axvline(syllable_end, c='k', linestyle='dashed', linewidth=3.0)
    plt.xlabel('Time (ms)')
    plt.ylabel('E%d (z-scored)' % e1)
    plt.axis('tight')
    plt.xlim(0, last_syllable_end)

    ax = plt.subplot(gs[2, :60])
    for k in range(ntrials_to_plot):
        plt.plot(t, lfp2[k, :], '-', linewidth=2.0, alpha=0.75)
    plt.plot(t, lfp2.mean(axis=0), 'k-', linewidth=3.0)
    plt.axvline(syllable_start, c='k', linestyle='dashed', linewidth=3.0)
    plt.axvline(syllable_end, c='k', linestyle='dashed', linewidth=3.0)
    plt.xlabel('Time (ms)')
    plt.ylabel('E%d (z-scored)' % e2)
    plt.axis('tight')
    plt.xlim(0, last_syllable_end)

    # restrict the lfps to a single syllable
    print 'syllable_start=%f, syllable_end=%f' % (syllable_start, syllable_end)
    syllable_si = int(syllable_start*sample_rate)
    syllable_ei = int(syllable_end*sample_rate)
    lfp1 = lfp1[:, syllable_si:syllable_ei]
    lfp2 = lfp2[:, syllable_si:syllable_ei]

    # compute the trial averaged and mean subtracted lfps
    lfp1_mean,lfp1_ms = compute_avg_and_ms(lfp1)
    lfp2_mean,lfp2_ms = compute_avg_and_ms(lfp2)

    psd_stats = get_psd_stats(bird, block, segment, hemi)
    freqs, psd1, psd2, psd1_ms, psd2_ms, c12, c12_nonlocked, c12_total = compute_spectra_and_coherence_single_electrode(lfp1, lfp2, sample_rate, e1, e2, psd_stats=psd_stats)
    lags_ms = get_lags_ms(sample_rate)

    lfp_absmax = max(np.abs(lfp1_mean).max(), np.abs(lfp2_mean).max())
    lfp_ms_absmax = max(np.abs(lfp1_ms).max(), np.abs(lfp2_ms).max())

    ax = plt.subplot(gs[0, 65:80])
    plt.axhline(0, c='k')
    plt.plot(t[syllable_si:syllable_ei], lfp1_mean, 'k-', linewidth=3.0, alpha=1.)
    plt.plot(t[syllable_si:syllable_ei], lfp2_mean, '-', c='#c0c0c0', linewidth=3.0)
    # plt.xlabel('Time (s)')
    plt.ylabel('Trial-avg LFP')
    leg = custom_legend(['k', '#c0c0c0'], ['E%d' % e1, 'E%d' % e2])
    plt.legend(handles=leg, fontsize='x-small')
    plt.axis('tight')
    plt.ylim(-lfp_absmax, lfp_absmax)

    ax = plt.subplot(gs[0, 85:])
    plt.axhline(0, c='k')
    plt.plot(t[syllable_si:syllable_ei], lfp1_ms, 'k-', linewidth=3.0, alpha=1.)
    plt.plot(t[syllable_si:syllable_ei], lfp2_ms, '-', c='#c0c0c0', linewidth=3.0)
    # plt.xlabel('Time (s)')
    plt.ylabel('Mean-sub LFP')
    plt.legend(handles=leg, fontsize='x-small')
    plt.axis('tight')
    plt.ylim(-lfp_ms_absmax, lfp_ms_absmax)

    psd_max = max(psd1.max(), psd2.max(), psd1_ms.max(), psd2_ms.max())

    ax = plt.subplot(gs[1, 65:80])
    plt.axhline(0, c='k')
    plt.plot(freqs, psd1, 'k-', linewidth=3.0)
    plt.plot(freqs, psd2, '-', c='#c0c0c0', linewidth=3.0)
    plt.xlabel('Time (s)')
    plt.ylabel('Trial-avg Power')
    plt.legend(handles=leg, fontsize='x-small', loc=2)
    plt.axis('tight')
    # plt.ylim(0, psd_max)

    ax = plt.subplot(gs[1, 85:])
    plt.axhline(0, c='k')
    plt.plot(freqs, psd1_ms, 'k-', linewidth=3.0)
    plt.plot(freqs, psd2_ms, '-', c='#c0c0c0', linewidth=3.0)
    plt.xlabel('Time (s)')
    plt.ylabel('Mean-sub Power')
    plt.legend(handles=leg, fontsize='x-small', loc=2)
    plt.axis('tight')
    # plt.ylim(0, psd_max)

    ax = plt.subplot(gs[2, 65:80])
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.plot(lags_ms, c12_total, 'k-', linewidth=3.0, alpha=0.75)
    plt.plot(lags_ms, c12, '-', c='r', linewidth=3.0, alpha=0.75)
    plt.plot(lags_ms, c12_nonlocked, '-', c='b', linewidth=3.0, alpha=0.75)
    plt.xlabel('Lags (ms)')
    plt.ylabel('Coherency')
    leg = custom_legend(['k', 'r', 'b'], ['Raw', 'Trial-avg', 'Mean-sub'])
    plt.legend(handles=leg, fontsize='x-small')
    plt.axis('tight')
    plt.ylim(-0.2, 0.3)

    fname = os.path.join(get_this_dir(), 'raw+coherency.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':

    spec_colormap()
    set_font()

    bird = 'GreBlu9508M'
    # bird = 'YelBlu6903F'
    block = 'Site4'
    segment = 'Call1'
    hemi = 'L'
    # hemi = 'R'
    e1 = 13
    e2 = 5
    # e1 = 24
    # e2 = 23
    stim_id = 112
    syllable_index = 1

    draw_figures(bird, block, segment, hemi, e1, e2, stim_id, syllable_index)
    plt.show()
