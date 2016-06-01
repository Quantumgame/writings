import os

import matplotlib.pyplot as plt

from lasp.sound import spec_colormap
from utils import get_this_dir, set_font, get_freqs, compute_spectra_and_coherence_single_electrode, get_psd_stats, \
    get_lags_ms
from zeebeez.transforms.stim_event import StimEventTransform
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def plot_cross_pair(spectra, cfs, e1, e2, freqs, lags_ms):

    nelectrodes = 2

    # make a plot of stim-locked covariances
    figsize = (16.0, 9)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.93, hspace=0.20, wspace=0.20)
    clrs = ['k']
    alphas = [0.9]
    gs = plt.GridSpec(nelectrodes, nelectrodes)

    electrodes = [e1, e2]

    # plot upper left
    ax = plt.subplot(2, 2, 1)
    plt.axhline(0, c='k')
    plt.plot(freqs, spectra[e1], 'k-', linewidth=4.0, alpha=1)
    plt.ylabel('Power (z-scored)')
    plt.axis('tight')
    plt.ylim(-1, 3)

    # plot lower right
    ax = plt.subplot(2, 2, 4)
    plt.axhline(0, c='k')
    plt.plot(freqs, spectra[e2], 'k-', linewidth=4.0, alpha=1)
    plt.ylabel('Power (z-scored)')
    plt.axis('tight')
    plt.ylim(-1, 3)

    # plot lower left
    ax = plt.subplot(2, 2, 3)
    plt.axhline(0, c='k')
    plt.axvline(0, c='k')
    plt.plot(lags_ms, cfs[(e1,e2)], 'k-', linewidth=4.0, alpha=1)
    plt.ylabel('Coherency')
    plt.axis('tight')
    plt.ylim(-0.2, 0.3)


def draw_figures(stim_event, stim_id, syllable_index):

    assert isinstance(stim_event, StimEventTransform)

    sample_rate = stim_event.lfp_sample_rate

    specs = dict()
    syllable_times = dict()
    stim_end_time = dict()

    hemi = ','.join(stim_event.rcg_names)

    # compute all cross and auto-correlations
    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

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

    # compute the covariance functions
    freqs = get_freqs(stim_event.lfp_sample_rate)
    lags_ms = get_lags_ms(stim_event.lfp_sample_rate)
    spectra,cfs = compute_stim_avg_cf(stim_event, lfp, electrode_order)
    e1,e2 = cfs.keys()[0]
    plot_cross_pair(spectra, cfs, e1, e2, freqs, lags_ms)

    fname = os.path.join(get_this_dir(), 'cross_total.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def compute_stim_avg_cf(stim_event, lfp, electrode_order):

    electrode_indices = stim_event.index2electrode
    sr = stim_event.lfp_sample_rate

    ntrials,nelectrodes,nt = lfp.shape

    seg_uname = stim_event.seg_uname
    bird,block,seg = seg_uname.split('_')
    hemi = stim_event.rcg_names[0]
    psd_stats = get_psd_stats(bird, stim_event.block_name, stim_event.segment_name, hemi)

    spectra = dict()
    cfs = dict()

    for i in range(nelectrodes):
        for j in range(i):

            lfp1 = lfp[:, i, :]
            lfp2 = lfp[:, j, :]

            _e1 = electrode_indices[i]
            _e2 = electrode_indices[j]

            pfreq, psd1, psd2, psd1_ms, psd2_ms, c12, c12_nonlocked, c12_total = compute_spectra_and_coherence_single_electrode(lfp1, lfp2,
                                                                                                              sr, _e1, _e2,
                                                                                                              psd_stats=psd_stats)

            spectra[_e1] = psd1
            spectra[_e2] = psd2
            cfs[(_e1, _e2)] = c12

    return spectra,cfs


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
    syllable_index = 2

    file_ext = '%s_%s_%s_%s' % (bird, block, segment, hemi)
    data_dir = '/auto/tdrive/mschachter/data/%s' % bird
    output_dir = os.path.join(data_dir, 'transforms')

    sefile = os.path.join(output_dir, 'StimEvent_%s.h5' % file_ext)
    se = StimEventTransform.load(sefile, rep_types_to_load=['raw'])
    se.segment_stims()
    se.preprocess('raw')

    draw_figures(se, stim_id, syllable_index)
    plt.show()
