import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from lasp.spikes import compute_psth

from lasp.colormaps import viridis, inferno

from lasp.sound import plot_spectrogram

from zeebeez.transforms.biosound import BiosoundTransform
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform
from zeebeez.transforms.stim_event import StimEventTransform
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def get_full_data(bird, block, segment, hemi, stim_id, data_dir='/auto/tdrive/mschachter/data'):

    bdir = os.path.join(data_dir, bird)
    tdir = os.path.join(bdir, 'transforms')

    aprops = ['meanspect', 'q1', 'q2', 'q3', 'sal', 'maxAmp', 'meantime', 'stdtime']

    # load the BioSound
    bs_file = os.path.join(tdir, 'BiosoundTransform_%s.h5' % bird)
    bs = BiosoundTransform.load(bs_file)

    # load the StimEvent transform
    se_file = os.path.join(tdir, 'StimEvent_%s_%s_%s_%s.h5' % (bird,block,segment,hemi))
    print 'Loading %s...' % se_file
    se = StimEventTransform.load(se_file, rep_types_to_load=['raw'])
    se.zscore('raw')
    se.segment_stims_from_biosound(bs_file)

    # load the pairwise CF transform
    pcf_file = os.path.join(tdir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird,block,segment,hemi))
    print 'Loading %s...' % pcf_file
    pcf = PairwiseCFTransform.load(pcf_file)

    def log_transform(x, dbnoise=100.):
        x /= x.max()
        zi = x > 0
        x[zi] = 20*np.log10(x[zi]) + dbnoise
        x[x < 0] = 0
        x /= x.max()

    all_lfp_psds = deepcopy(pcf.psds)
    log_transform(all_lfp_psds)
    all_lfp_psds -= all_lfp_psds.mean(axis=0)
    all_lfp_psds /= all_lfp_psds.std(axis=0, ddof=1)

    all_spike_psds = deepcopy(pcf.spike_psd)
    log_transform(all_spike_psds, dbnoise=80.)
    all_spike_psds -= all_spike_psds.mean(axis=0)
    all_spike_psds /= all_spike_psds.std(axis=0, ddof=1)

    # get overall biosound stats
    bs_stats = dict()
    for aprop in aprops:
        amean = bs.stim_df[aprop].mean()
        astd = bs.stim_df[aprop].std(ddof=1)
        bs_stats[aprop] = (amean, astd)

    # get the spectrogram
    i = se.segment_df.stim_id == stim_id
    last_end_time = se.segment_df.end_time[i].max()

    spec_freq = se.spec_freq
    stim_spec = se.spec_by_stim[stim_id]
    spec_t = np.arange(stim_spec.shape[1]) / se.lfp_sample_rate
    speci = np.min(np.where(spec_t > last_end_time)[0])
    spec_t = spec_t[:speci]
    stim_spec = stim_spec[:, :speci]
    stim_dur = spec_t.max() - spec_t.min()

    # get the raw LFP
    si = int(se.pre_stim_time*se.lfp_sample_rate)
    ei = int(stim_dur*se.lfp_sample_rate) + si
    lfp = se.lfp_reps_by_stim['raw'][stim_id][:, :, si:ei]
    ntrials,nelectrodes,nt = lfp.shape

    # get the raw spikes, spike_mat is ragged array of shape (num_trials, num_cells, num_spikes)
    spike_mat = se.spikes_by_stim[stim_id]
    assert ntrials == len(spike_mat)

    ncells = len(se.cell_df)
    print 'ncells=%d' % ncells
    ntrials = len(spike_mat)

    # compute the PSTH
    psth = list()
    for n in range(ncells):
        # get the spikes across all trials for neuron n
        spikes = [spike_mat[k][n] for k in range(ntrials)]
        # make a PSTH
        _psth_t,_psth = compute_psth(spikes, stim_dur, bin_size=1.0/se.lfp_sample_rate)
        psth.append(_psth)
    psth = np.array(psth)

    if hemi == 'L':
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
    else:
        electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

    # get acoustic props and LFP/spike power spectra for each syllable
    syllable_props = list()

    i = bs.stim_df.stim_id == stim_id
    orders = sorted(bs.stim_df.order[i].values)
    cell_index2electrode = None
    for o in orders:
        i = (bs.stim_df.stim_id == stim_id) & (bs.stim_df.order == o)
        assert i.sum() == 1

        d = dict()
        d['start_time'] = bs.stim_df.start_time[i].values[0]
        d['end_time'] = bs.stim_df.end_time[i].values[0]
        d['order'] = o

        for aprop in aprops:
            amean,astd = bs_stats[aprop]
            d[aprop] = (bs.stim_df[aprop][i].values[0] - amean) / astd

        # get the LFP power spectra
        lfp_psd = list()
        for k,e in enumerate(electrode_order):
            i = (pcf.df.stim_id == stim_id) & (pcf.df.order == o) & (pcf.df.decomp == 'locked') & \
                (pcf.df.electrode1 == e) & (pcf.df.electrode2 == e)

            assert i.sum() == 1

            index = pcf.df[i]['index'].values[0]
            lfp_psd.append(all_lfp_psds[index, :])
        d['lfp_psd'] = np.array(lfp_psd)

        # get the PSTH power spectra
        psth_psd = list()

        cell_i2e = list()
        for k,e in enumerate(electrode_order):
            i = (pcf.df.stim_id == stim_id) & (pcf.df.order == o) & (pcf.df.decomp == 'spike_psd') & \
                (pcf.df.electrode1 == e) & (pcf.df.electrode2 == e)
            assert i.sum() > 0, "i.sum()=%d" % i.sum()

            cindex = pcf.df.cell_index[i].values

            for ci in cindex:
                i = (pcf.df.stim_id == stim_id) & (pcf.df.order == o) & (pcf.df.decomp == 'spike_psd') & \
                    (pcf.df.electrode1 == e) & (pcf.df.electrode2 == e) & (pcf.df.cell_index == ci)

                assert i.sum() == 1, "i.sum()=%d" % i.sum()

                index = pcf.df['index'][i].values[0]
                psd = all_spike_psds[index, :]
                psth_psd.append(psd)

                cell_i2e.append(e)

        if cell_index2electrode is None:
            cell_index2electrode = cell_i2e

        d['psth_psd'] = psth_psd

        syllable_props.append(d)

    return {'stim_id':stim_id, 'spec_t':spec_t, 'spec_freq':spec_freq, 'spec':stim_spec,
            'lfp':lfp, 'spikes':spike_mat, 'lfp_sample_rate':se.lfp_sample_rate, 'psth':psth,
            'syllable_props':syllable_props, 'electrode_order':electrode_order, 'psd_freq':pcf.freqs,
            'cell_index2electrode':cell_index2electrode, 'aprops':aprops}


def plot_full_data(d, syllable_index):

    syllable_start = d['syllable_props'][syllable_index]['start_time'] - 0.020
    syllable_end = d['syllable_props'][syllable_index]['end_time'] + 0.030

    figsize = (24.0, 10)
    fig = plt.figure(figsize=figsize, facecolor='w')
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20, wspace=0.20)

    gs = plt.GridSpec(100, 100)
    left_width = 55
    top_height = 15
    middle_height = 30
    bottom_height = 45

    # plot the spectrogram
    ax = plt.subplot(gs[:top_height+1, :left_width])
    spec = d['spec']
    spec[spec < np.percentile(spec, 15)] = 0
    plot_spectrogram(d['spec_t'], d['spec_freq'], spec, ax=ax, colormap=plt.cm.afmhot_r, colorbar=False, ticks=True)
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)

    # plot the LFPs
    sr = d['lfp_sample_rate']
    lfp_mean = d['lfp'].mean(axis=0)
    lfp_t = np.arange(lfp_mean.shape[1]) / sr
    nelectrodes,nt = lfp_mean.shape
    gs_i = top_height + 5
    gs_e = gs_i + middle_height + 1

    ax = plt.subplot(gs[gs_i:gs_e, :left_width])

    voffset = 5
    for n in range(nelectrodes):
        plt.plot(lfp_t, lfp_mean[nelectrodes-n-1, :] + voffset*n, 'k-', linewidth=3.0, alpha=0.75)
    plt.axis('tight')
    ytick_locs = np.arange(nelectrodes) * voffset
    plt.yticks(ytick_locs, list(reversed(d['electrode_order'])))
    plt.ylabel('Trial-averaged LFP (Electrode)')
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)

    # plot the PSTH
    gs_i = gs_e + 5
    gs_e = gs_i + bottom_height + 1
    ax = plt.subplot(gs[gs_i:gs_e, :left_width])
    ncells = d['psth'].shape[0]
    plt.imshow(d['psth'], interpolation='nearest', aspect='auto', origin='upper', extent=(0, lfp_t.max(), ncells, 0),
               cmap=plt.cm.bone_r)

    cell_i2e = d['cell_index2electrode']
    print 'cell_i2e=',cell_i2e
    last_electrode = cell_i2e[0]
    for k,e in enumerate(cell_i2e):
        if e != last_electrode:
            plt.axhline(k, c='k', alpha=0.5)
            last_electrode = e

    ytick_locs = list()
    for e in d['electrode_order']:
        elocs = np.array([k for k,el in enumerate(cell_i2e) if el == e])
        emean = elocs.mean()
        ytick_locs.append(emean+0.5)
    plt.yticks(ytick_locs, d['electrode_order'])
    plt.ylabel('PSTH (Electrode)')

    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)

    # plot the biosound properties
    sprops = d['syllable_props'][syllable_index]
    aprops = d['aprops']

    vals = [sprops[a] for a in aprops]
    ax = plt.subplot(gs[:top_height, (left_width+5):])
    plt.axhline(0, c='k')
    plt.bar(range(len(aprops)), vals, color='#c0c0c0')
    plt.axis('tight')
    plt.ylim(-1.5, 1.5)
    plt.xticks(np.arange(len(aprops))+0.5, aprops)
    plt.ylabel('Z-score')

    # plot the LFP power spectra
    gs_i = top_height + 5
    gs_e = gs_i + middle_height + 1

    f = d['psd_freq']
    ax = plt.subplot(gs[gs_i:gs_e, (left_width+5):])
    plt.imshow(sprops['lfp_psd'], interpolation='nearest', aspect='auto', origin='upper',
               extent=(f.min(), f.max(), nelectrodes, 0), cmap=plt.cm.bwr)
    plt.xlabel('Frequency (Hz)')
    plt.yticks(np.arange(nelectrodes)+0.5, d['electrode_order'])

    # plot the PSTH power spectra
    gs_i = gs_e + 5
    gs_e = gs_i + bottom_height + 1

    ax = plt.subplot(gs[gs_i:gs_e, (left_width+5):])
    plt.imshow(sprops['psth_psd'], interpolation='nearest', aspect='auto', origin='upper',
               extent=(f.min(), f.max(), ncells, 0), cmap=plt.cm.bwr)
    plt.xlabel('Frequency (Hz)')


def draw_figures():

    d = get_full_data('GreBlu9508M', 'Site4', 'Call1', 'L', 268)
    plot_full_data(d, 1)

    plt.show()


if __name__ == '__main__':
    draw_figures()