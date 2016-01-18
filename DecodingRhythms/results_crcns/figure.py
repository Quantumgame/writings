import os
from copy import deepcopy

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import COLOR_YELLOW_SPIKE, get_this_dir, COLOR_BLUE_LFP
from lasp.sound import log_transform
from utils import set_font
from zeebeez.aggregators.pairwise_cf import AggregatePairwiseCF
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def plot_psds(psd_file, data_dir='/auto/tdrive/mschachter/data'):

    # read PairwiseCF file
    pcf_file = os.path.join(data_dir, 'aggregate', 'pairwise_cf.h5')
    pcf = AggregatePairwiseCF.load(pcf_file)
    pcf.zscore_within_site()

    # get the LFP PSDs
    indices = pcf.df['lfp_index'].unique()
    lfp_psds = pcf.lfp_psds[indices, :]
    psum = lfp_psds.sum(axis=1)
    nz = psum > 0
    lfp_psds = lfp_psds[nz, :]
    log_transform(lfp_psds)
    lfp_psd_mean = lfp_psds.mean(axis=0)
    lfp_psd_std = lfp_psds.std(axis=0, ddof=1)
    nsamps_lfp = lfp_psds.shape[0]

    # get the PSTH PSDs
    indices = pcf.df['spike_index'].unique()
    spike_psds = pcf.spike_psds[indices, :]
    psum = spike_psds.sum(axis=1)
    nz = psum > 0
    spike_psds = spike_psds[nz, :]
    log_transform(spike_psds)
    spike_psd_mean = spike_psds.mean(axis=0)
    spike_psd_std = spike_psds.std(axis=0, ddof=1)
    nsamps_spike = spike_psds.shape[0]

    # get the spike field coherence
    indices = pcf.df['spike_index'].unique()
    spike_field_coherences = pcf.spike_field_coherences[indices, :]
    psum = spike_field_coherences.sum(axis=1)
    nz = psum > 0
    spike_field_coherences = spike_field_coherences[nz, :]
    log_transform(spike_field_coherences)
    spike_field_coherences_mean = spike_field_coherences.mean(axis=0)
    spike_field_coherences_std = spike_field_coherences.std(axis=0, ddof=1)

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

    fig = plt.figure(figsize=(24, 5.5))
    fig.subplots_adjust(wspace=0.30, hspace=0.30)

    nrows = 1
    ncols = 3

    ax = plt.subplot(nrows, ncols, 1)
    plt.errorbar(pcf.freqs, lfp_psd_mean, yerr=lfp_psd_std/np.sqrt(nsamps_lfp), c=COLOR_BLUE_LFP, linewidth=4.0, elinewidth=2.0, ecolor='k', alpha=0.7)
    plt.errorbar(pcf.freqs+3, spike_psd_mean, yerr=spike_psd_std/np.sqrt(nsamps_spike), c=COLOR_YELLOW_SPIKE, linewidth=4.0, elinewidth=2.0, ecolor='k', alpha=0.7)
    plt.legend(['LFP', 'Spike PSD'], fontsize='x-small')
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.ylim(0, 1)

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
    plt.legend(['MLd', 'OV', 'Field L+CM'], fontsize='x-small', loc='lower right')

    ax = plt.subplot(nrows, ncols, 3)
    fi = pcf.freqs > 20
    spike_field_coherences_mean[~fi] = 0
    plt.errorbar(pcf.freqs, spike_field_coherences_mean, yerr=spike_field_coherences_std/np.sqrt(nsamps_spike), c='k', linewidth=4.0,
                 alpha=0.7, elinewidth=2., ecolor='k')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('PSTH-LFP Coherence')
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