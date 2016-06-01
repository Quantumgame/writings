import os

import matplotlib.pyplot as plt
import numpy as np

from lasp.sound import plot_spectrogram, spec_colormap
from utils import get_this_dir, set_font, compute_spectra_and_coherence_single_electrode, get_lags_ms, get_psd_stats
from zeebeez.transforms.stim_event import StimEventTransform
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def plot_cross_pair(stim_ids, spectra_and_coherence_by_stim, electrode_order, freqs, lags_ms):

    # make a plot of stim-locked covariances
    figsize = (16.0, 9)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.93, hspace=0.20, wspace=0.20)
    clrs = ['b', 'g', 'r']
    alphas = [0.6, 0.75, 0.65]

    low_freq = -1
    high_freq = 3

    ax = plt.subplot(2, 2, 1)
    for k,stim_id in enumerate(stim_ids):
        psd1, psd2, c12 = spectra_and_coherence_by_stim[stim_id]
        clr = clrs[k]
        plt.axhline(0, c='k')
        plt.plot(freqs, psd1, '-', c=clr, linewidth=3.0, alpha=alphas[k])
        plt.axis('tight')
        plt.ylim(low_freq, high_freq)
        plt.ylabel('Power (z-scored)')
        plt.xlabel('Frequency (Hz)')

    ax = plt.subplot(2, 2, 3)
    for k,stim_id in enumerate(stim_ids):
        psd1, psd2, c12 = spectra_and_coherence_by_stim[stim_id]
        clr = clrs[k]
        plt.axhline(0, c='k')
        plt.axvline(0, c='k')
        plt.plot(lags_ms, c12, '-', c=clr, linewidth=3.0, alpha=alphas[k])
        plt.axis('tight')
        plt.ylim(-0.2, 0.3)
        plt.ylabel('Coherency')
        plt.xlabel('Lag (ms)')

    ax = plt.subplot(2, 2, 4)
    for k,stim_id in enumerate(stim_ids):
        psd1, psd2, c12 = spectra_and_coherence_by_stim[stim_id]
        clr = clrs[k]
        plt.axhline(0, c='k')
        plt.plot(freqs, psd2, '-', c=clr, linewidth=3.0, alpha=alphas[k])
        plt.axis('tight')
        plt.ylim(low_freq, high_freq)
        plt.ylabel('Power (z-scored)')
        plt.xlabel('Frequency (Hz)')


def draw_figures(stim_event, stim_ids, syllable_indices, e1=5, e2=2):

    assert isinstance(stim_event, StimEventTransform)

    sample_rate = stim_event.lfp_sample_rate
    lags_ms = get_lags_ms(sample_rate)

    cross_functions_total = dict()
    cross_functions_locked = dict()
    cross_functions_nonlocked = dict()
    specs = dict()
    syllable_times = dict()
    stim_end_time = dict()

    hemi = ','.join(stim_event.rcg_names)

    # compute all cross and auto-correlations
    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    freqs = None
    nelectrodes = None
    index2electrode = stim_event.index2electrode

    seg_uname = stim_event.seg_uname
    bird,block,seg = seg_uname.split('_')
    psd_stats = get_psd_stats(bird, stim_event.block_name, stim_event.segment_name, hemi)

    for stim_id,syllable_index in zip(stim_ids, syllable_indices):
        lfp = stim_event.lfp_reps_by_stim['raw'][stim_id]
        ntrials,nelectrodes,nt = lfp.shape

        # get the start and end time of the syllable
        i = (stim_event.segment_df['stim_id'] == stim_id) & (stim_event.segment_df['order'] == syllable_index)
        assert i.sum() > 0, "No syllable for stim_id=%d, order=%d" % (stim_id, syllable_index)
        assert i.sum() == 1, "More than one syllable for stim_id=%d, order=%d" % (stim_id, syllable_index)
        start_time = stim_event.segment_df[i]['start_time'].values[0]
        end_time = stim_event.segment_df[i]['end_time'].values[0]
        syllable_times[stim_id] = (start_time, end_time)

        # get the end time of the last syllable
        i = (stim_event.segment_df['stim_id'] == stim_id)
        stim_end_time[stim_id] = stim_event.segment_df[i]['end_time'].max()

        si = int((stim_event.pre_stim_time + start_time)*sample_rate)
        ei = int((stim_event.pre_stim_time + end_time)*sample_rate)

        # restrict the lfp to just the syllable time
        lfp = lfp[:, :, si:ei]
        specs[stim_id] = stim_event.spec_by_stim[stim_id]

        i1 = index2electrode.index(e1)
        lfp1 = lfp[:, i1, :]

        i2 = index2electrode.index(e2)
        lfp2 = lfp[:, i2, :]

        # compute the covariance functions
        pfreq, psd1, psd2, psd1_ms, psd2_ms, c12, c12_nonlocked, c12_total = compute_spectra_and_coherence_single_electrode(lfp1, lfp2,
                                                                                                          sample_rate,
                                                                                                          e1, e2,
                                                                                                          psd_stats=psd_stats)
        cross_functions_total[stim_id] = (psd1, psd2, c12_total)
        cross_functions_locked[stim_id] = (psd1, psd2, c12)
        cross_functions_nonlocked[stim_id] = (psd1, psd2, c12_nonlocked)

    # plot the cross covariance functions to overlap for each stim
    plot_cross_pair(stim_ids, cross_functions_total, electrode_order, freqs, lags_ms)
    plt.suptitle('Total Covariance')
    fname = os.path.join(get_this_dir(), 'cross_total.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    # plot_cross_mat(stim_ids, cross_functions_locked, electrode_order, freqs)
    plot_cross_pair(stim_ids, cross_functions_locked, electrode_order, freqs, lags_ms)
    plt.suptitle('Stim-locked Covariance')
    fname = os.path.join(get_this_dir(), 'cross_locked.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    # plot_cross_mat(stim_ids, cross_functions_nonlocked, electrode_order, freqs)
    plot_cross_pair(stim_ids, cross_functions_nonlocked, electrode_order, freqs, lags_ms)
    plt.suptitle('Non-locked Covariance')
    fname = os.path.join(get_this_dir(), 'cross_nonlocked.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    # plot the spectrograms
    fig_height = 2
    fig_max_width = 6
    spec_lens = np.array([stim_end_time[stim_id] for stim_id in stim_ids])
    spec_ratios = spec_lens / spec_lens.max()
    spec_sample_rate = sample_rate

    for k,stim_id in enumerate(stim_ids):
        spec = specs[stim_id]
        syllable_start,syllable_end = syllable_times[stim_id]
        print 'stim_id=%d, syllable_start=%0.3f, syllable_end=%0.3f' % (stim_id, syllable_start, syllable_end)
        spec_t = np.arange(spec.shape[1]) / spec_sample_rate
        stim_end = stim_end_time[stim_id]

        figsize = (spec_ratios[k]*fig_max_width, fig_height)
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        plot_spectrogram(spec_t, stim_event.spec_freq, spec, ax=ax, ticks=True, fmin=300., fmax=8000.,
                         colormap='SpectroColorMap', colorbar=False)
        plt.axvline(syllable_start, c='k', linestyle='dashed', linewidth=3.0)
        plt.axvline(syllable_end, c='k', linestyle='dashed', linewidth=3.0)
        plt.xlim(0, stim_end+0.005)
        fname = os.path.join(get_this_dir(), 'stim_spec_%d.svg' % stim_id)
        # plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':

    spec_colormap()
    set_font()

    bird = 'GreBlu9508M'
    block = 'Site4'
    segment = 'Call1'
    hemi = 'L'
    e1 = 6
    e2 = 4
    stim_ids = [43, 53, 190]
    syllable_indices = [2, 1, 0]

    file_ext = '%s_%s_%s_%s' % (bird, block, segment, hemi)
    data_dir = '/auto/tdrive/mschachter/data/%s' % bird
    output_dir = os.path.join(data_dir, 'transforms')

    sefile = os.path.join(output_dir, 'StimEvent_%s.h5' % file_ext)
    se = StimEventTransform.load(sefile, rep_types_to_load=['raw'])
    se.segment_stims()
    se.preprocess('raw')

    draw_figures(se, stim_ids, syllable_indices)
    plt.show()
