import os
from copy import deepcopy

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import COLOR_YELLOW_SPIKE, get_this_dir, COLOR_BLUE_LFP
from lasp.sound import log_transform
from utils import set_font
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def plot_psds(psd_file, data_dir='/auto/tdrive/mschachter/data'):

    # read PairwiseCF file
    pcf_file = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    pcf = PairwiseCFTransform.load(pcf_file)

    # get the LFP PSDs
    i = (pcf.df.decomp == 'locked') & (pcf.df.electrode1 == pcf.df.electrode2)
    indices = pcf.df['index'][i].values
    lfp_psds = pcf.psds[indices, :]
    psum = lfp_psds.sum(axis=1)
    nz = psum > 0
    lfp_psds = lfp_psds[nz, :]
    log_transform(lfp_psds)
    lfp_psd_mean = lfp_psds.mean(axis=0)
    lfp_psd_std = lfp_psds.std(axis=0, ddof=1)

    # get the PSTH PSDs
    i = (pcf.df.decomp == 'spike_psd') & (pcf.df.electrode1 == pcf.df.electrode2)
    indices = pcf.df['index'][i].values
    spike_psds = pcf.spike_psd[indices, :]
    psum = spike_psds.sum(axis=1)
    nz = psum > 0
    spike_psds = spike_psds[nz, :]
    log_transform(spike_psds)
    spike_psd_mean = spike_psds.mean(axis=0)
    spike_psd_std = spike_psds.std(axis=0, ddof=1)

    # get the spike rates
    spike_rates = pcf.spike_rate[indices, :]
    spike_rates = spike_rates[nz, :]

    # get the spike field coherence
    i = (pcf.df.decomp == 'spike_field_coherence') & (pcf.df.electrode1 == pcf.df.electrode2)
    indices = pcf.df['index'][i].values
    spike_field_coherence = pcf.spike_field_coherence[indices, :]
    psum = spike_field_coherence.sum(axis=1)
    nz = psum > 0
    spike_field_coherence = spike_field_coherence[nz, :]
    sfc_mean = spike_field_coherence.mean(axis=0)
    sfc_std = spike_field_coherence.std(axis=0, ddof=1)

    # read CRCNS file
    cell_data = dict()
    hf = h5py.File(psd_file, 'r')
    cnames = hf.attrs['col_names']
    for c in cnames:
        cell_data[c] = np.array(hf[c])
    crcns_psds = np.array(hf['psds'])
    freqs = hf.attrs['freqs']
    hf.close()

    cell_df = pd.DataFrame(cell_data)
    print 'regions=',cell_df.superregion.unique()

    name_map = {'brainstem':'MLd', 'thalamus':'OV', 'cortex':'Field L+CM'}

    # compute low gamma center frequencies
    fi_gamma = (freqs > 0) & (freqs < 100) # 16-66Hz

    # draw the spike rate vs each frequency band power
    """
    nrows = 2
    ncols = 6
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.40, hspace=0.40)
    for k,f in enumerate(pcf.freqs):
        ax = plt.subplot(nrows, ncols, k+1)
        x = spike_psds[:, k]
        r = spike_rates[:, 0]

        xnz = (x > 0) & (r > 0)
        x = x[xnz]
        x -= x.mean()
        x /= x.std(ddof=1)

        r = np.log2(r[xnz])
        cc = np.corrcoef(x, r)[0, 1]

        plt.plot(x, r, 'go')
        plt.xlabel('Z-scored Power at %dHz' % f)
        plt.ylabel('Log Spike Rate')
        plt.title('cc=%0.2f' % cc)
        plt.axis('tight')
    """

    # draw figure
    fig = plt.figure(figsize=(24, 8))
    fig.subplots_adjust(wspace=0.30, hspace=0.30)

    nrows = 1
    ncols = 3

    # plot the grand mean PSD for LFP and PSTH
    ax = plt.subplot(nrows, ncols, 1)
    plt.plot(pcf.freqs, lfp_psd_mean, '-', c=COLOR_BLUE_LFP, linewidth=4.0, alpha=0.7)
    plt.plot(pcf.freqs, spike_psd_mean, '-', c=COLOR_YELLOW_SPIKE, linewidth=4.0, alpha=0.7)
    plt.legend(['LFP', 'Spike PSD'])
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')

    # plot the grand mean PSD for PSTH using the CRCNS data
    fi = freqs < 200
    ax = plt.subplot(nrows, ncols, 2)
    clrs = ['r', 'g', 'b']
    for k,reg in enumerate(['brainstem', 'thalamus', 'cortex']):

        i = cell_df.superregion == reg
        indices = cell_df['index'][i].values
        psds = crcns_psds[indices, :]
        log_psds = deepcopy(psds)
        log_transform(log_psds)

        # compute the mean and sd of the power spectra
        psd_mean = log_psds.mean(axis=0)
        psd_std = log_psds.std(axis=0, ddof=1)
        psd_cv = psd_std / psd_mean

        # plot the mean power spectrum on the left

        plt.plot(freqs[fi], psd_mean[fi], c=clrs[k], linewidth=4.0, alpha=0.7)
        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.axis('tight')
        # plt.ylim(0, 0.8)
    plt.legend(['MLd', 'OV', 'Field L+CM'])

    # plot the average spike field coherence
    ax = plt.subplot(nrows, ncols, 3)
    plt.plot(pcf.freqs, sfc_mean, 'k-', linewidth=4.0, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSTH-LFP Coherence')
    plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'crcns_data.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

def draw_figures():
    crcns_dir='/auto/tdrive/mschachter/data/crcns'
    psd_file = os.path.join(crcns_dir, 'cell_psd.h5')
    plot_psds(psd_file)

    plt.show()


if __name__ == '__main__':
    set_font()
    draw_figures()