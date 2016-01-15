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
    spike_psds = pcf.psds[indices, :]
    psum = lfp_psds.sum(axis=1)
    nz = psum > 0
    spike_psds = spike_psds[nz, :]
    log_transform(spike_psds)
    spike_psd_mean = spike_psds.mean(axis=0)
    spike_psd_std = spike_psds.std(axis=0, ddof=1)

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

    print 'all_psds.shape=',crcns_psds.shape

    """
    fig = plt.figure(figsize=(24, 8))
    fig.subplots_adjust(wspace=0.30, hspace=0.30)
    nrows = 2
    ncols = 3
    gs = plt.GridSpec(nrows, ncols)
    for k,reg in enumerate(['brainstem', 'thalamus', 'cortex']):

        i = cell_df.superregion == reg
        indices = cell_df['index'][i].values
        psds = all_psds[indices, :]
        log_psds = deepcopy(psds)
        log_transform(log_psds)

        # compute the mean and sd of the power spectra
        psd_mean = log_psds.mean(axis=0)
        psd_std = log_psds.std(axis=0, ddof=1)

        # normalize psds to compute center frequencies
        psds = psds[:, fi_gamma]
        psds = (psds.T / psds.sum(axis=1)).T

        psd_sum = psds.sum(axis=1)
        ni = ~np.isnan(psd_sum)

        psds = psds[ni, :]
        cfreq = np.dot(psds, freqs[fi_gamma])
        nsamps = psds.shape[0]
        print '%s: nsamps=%d' % (reg, nsamps)

        # plot the mean power spectrum on the left
        ax = plt.subplot(gs[0, k])
        fi = freqs < 200
        plt.errorbar(freqs[fi], psd_mean[fi], yerr=psd_std[fi], c='k', linewidth=4.0, alpha=0.7, elinewidth=2.0)
        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.axis('tight')
        plt.title(name_map[reg])
        plt.ylim(0, 0.8)

        ax = plt.subplot(gs[1, k])
        plt.hist(cfreq, bins=7, color=COLOR_YELLOW_SPIKE, normed=True)
        plt.axis('tight')
        plt.xlabel('Center Freq (Hz)')
        plt.ylabel('Proportion')
        plt.ylim(0, 0.18)
        plt.xlim(15, 100)
    """



    fig = plt.figure(figsize=(24, 8))
    fig.subplots_adjust(wspace=0.30, hspace=0.30)

    ncols = 2

    ax = plt.subplot(1, ncols, 1)
    plt.plot(pcf.freqs, lfp_psd_mean, '-', c=COLOR_BLUE_LFP, linewidth=4.0, alpha=0.7)
    plt.plot(pcf.freqs, spike_psd_mean, '-', c=COLOR_YELLOW_SPIKE, linewidth=4.0, alpha=0.7)
    plt.legend(['LFP', 'Spike PSD'])
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')

    fi = freqs < 200
    ax = plt.subplot(1, ncols, 2)
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