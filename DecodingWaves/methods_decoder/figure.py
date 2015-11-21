import os
import operator

import numpy as np

import matplotlib.pyplot as plt

from lasp.signal import break_envelope_into_events, correlation_function, coherency
from utils import get_this_dir, set_font
from zeebeez.aggregators.pairwise_decoders_single import AggregatePairwiseDecoder

from zeebeez.core.experiment import Experiment
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_RIGHT, ROSTRAL_CAUDAL_ELECTRODES_LEFT, CALL_TYPES, \
    CALL_TYPE_SHORT_NAMES, DECODER_CALL_TYPES


def draw_figures(bird, block, segment, hemi, stim_id, trial, syllable_index):
    draw_coherency_matrix(bird, block, segment, hemi, stim_id, trial, syllable_index)
    plot_confidence_matrix(bird, block, segment, hemi)


def draw_coherency_matrix(bird, block, segment, hemi, stim_id, trial, syllable_index,
                          data_dir='/auto/tdrive/mschachter/data', exp=None, save=True):

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
    lfp_data = exp.get_lfp_slice(seg, start_time, end_time)
    electrode_indices,lfps,sample_rate = lfp_data[hemi]

    # rescale the LFPs to they are in uV
    lfps *= 1e6

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
    syllable_si = int(syllable_start*sample_rate)
    syllable_ei = int(syllable_end*sample_rate)

    # compute all cross and auto-correlations
    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    lags = np.arange(-20, 21)
    lags_ms = (lags / sample_rate)*1e3

    window_fraction = 0.35
    noise_db = 25.
    nelectrodes = len(electrode_order)
    cross_mat = np.zeros([nelectrodes, nelectrodes, len(lags)])
    for i in range(nelectrodes):
        for j in range(nelectrodes):

            lfp1 = lfps[i, syllable_si:syllable_ei]
            lfp2 = lfps[j, syllable_si:syllable_ei]

            if i != j:
                x = coherency(lfp1, lfp2, lags, window_fraction=window_fraction, noise_floor_db=noise_db)
            else:
                x = correlation_function(lfp1, lfp2, lags)

            _e1 = electrode_indices[i]
            _e2 = electrode_indices[j]

            i1 = electrode_order.index(_e1)
            i2 = electrode_order.index(_e2)

            # print 'i=%d, j=%d, e1=%d, e2=%d, i1=%d, i2=%d' % (i, j, _e1, _e2, i1, i2)

            cross_mat[i1, i2, :] = x

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

            plt.plot(lags_ms, cross_mat[i, j, :], 'k-', linewidth=2.0)
            plt.xticks([])
            plt.yticks([])
            plt.axis('tight')
            plt.ylim(-0.5, 1.0)
            if j == 0:
                plt.ylabel('E%d' % electrode_order[i])
            if i == nelectrodes-1:
                plt.xlabel('E%d' % electrode_order[j])

    if save:
        fname = os.path.join(get_this_dir(), 'coherency_matrix.svg')
        plt.savefig(fname, facecolor='w', edgecolor='none')


def plot_confidence_matrix(bird, block, segment, hemi, e1=5, e2=4,
                           pfile='/auto/tdrive/mschachter/data/aggregate/decoders_pairwise_coherence_single.h5'):

    # read the confusion matrix for this decoder
    pcf = AggregatePairwiseDecoder.load(pfile)

    i = (pcf.df['bird'] == bird) & (pcf.df['block'] == block) & (pcf.df['segment'] == segment) & (pcf.df['hemi'] == hemi) & \
        (pcf.df['order'] == 'self+cross') & (pcf.df['e1'] == e1) & (pcf.df['e2'] == e2) & (pcf.df['decomp'] == 'locked')

    assert i.sum() == 1, "More than one model (i.sum()=%d)" % i.sum()

    index = pcf.df['index'][i].values[0]

    print 'confidence_matrices.keys()=',pcf.confidence_matrices.keys()

    # shuffle the entries in the confidence matrix around
    key = ('locked', 'self+cross', 'pair')
    cnames = list(pcf.class_names[key][index])
    print 'cnames=',cnames
    cmats = pcf.confidence_matrices[key]
    print 'cmats.shape=',cmats.shape
    C1 = cmats[index]

    C = np.zeros_like(C1)
    for i,cname1 in enumerate(cnames):
        for j,cname2 in enumerate(cnames):
            ii = DECODER_CALL_TYPES.index(cname1)
            jj = DECODER_CALL_TYPES.index(cname2)
            C[ii, jj] = C1[i, j]

    tick_labels = [CALL_TYPE_SHORT_NAMES[ct] for ct in DECODER_CALL_TYPES]
    # plot the confidence matrix
    figsize = (15.5, 11)
    fig = plt.figure(figsize=figsize)
    plt.imshow(C, origin='lower', interpolation='nearest', aspect='auto', vmin=0, vmax=1, cmap=plt.cm.afmhot)
    plt.xticks(range(len(CALL_TYPES)), tick_labels)
    plt.yticks(range(len(CALL_TYPES)), tick_labels)
    plt.axis('tight')
    plt.colorbar()
    plt.title('PCC=%0.2f' % (np.diag(C).mean()))

    fname = os.path.join(get_this_dir(), 'confidence_matrix.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':

    set_font()

    """
    bird = 'YelBlu6903F'
    block = 'Site2'
    segment = 'Call2'
    hemi = 'R'
    stim_id = 254
    trial = 2
    syllable_index = 1
    """

    bird = 'GreBlu9508M'
    block = 'Site3'
    segment = 'Call3'
    hemi = 'L'
    e1 = 5
    e2 = 4

    """
    bird = 'YelBlu6903F'
    block = 'Site3'
    segment = 'Call3'
    hemi = 'R'
    e1 = 30
    e2 = 28
    """

    # draw_figures(bird, block, segment, hemi, stim_id, trial, syllable_index)
    # draw_coherency_matrix(bird, block, segment, hemi, stim_id, trial, syllable_index, save=False)
    plot_confidence_matrix(bird, block, segment, hemi, e1=e1, e2=e2)
    plt.show()
