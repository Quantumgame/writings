import os
from copy import deepcopy

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasp.colormaps import magma
from sklearn.covariance import LedoitWolf

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

    g = pcf.df.groupby(['bird', 'block', 'segment', 'electrode'])
    nsamps_electrodes = len(g)

    i = pcf.df.cell_index != -1
    g = pcf.df[i].groupby(['bird', 'block', 'segment', 'electrode', 'cell_index'])
    nsamps_cells = len(g)

    print '# of electrodes: %d' % nsamps_electrodes
    print '# of cells: %d' % nsamps_cells
    print '# of lfp samples: %d' % (pcf.lfp_psds.shape[0])
    print '# of spike psd samples: %d' % (pcf.spike_psds.shape[0])
    return

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

    # zscore the LFP
    lfp_psds_z = deepcopy(lfp_psds)
    lfp_psds_z -= lfp_psds.mean(axis=0)
    lfp_psds_z /= lfp_psds.std(axis=0, ddof=1)

    # compute the LFP PSD covariance
    # lfp_cov_est = LedoitWolf()
    # lfp_cov_est.fit(lfp_psds_z)
    # lfp_cov = lfp_cov_est.covariance_

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
    
    # zscore the spike psds
    spike_psds_z = deepcopy(spike_psds)
    spike_psds_z -= spike_psds.mean(axis=0)
    spike_psds_z /= spike_psds.std(axis=0, ddof=1)

    # compute the spike PSD covariance
    # spike_cov_est = LedoitWolf()
    # spike_cov_est.fit(spike_psds_z)
    # spike_cov = spike_cov_est.covariance_

    # concatenate the lfp and spike psds
    nfreqs = len(pcf.freqs)
    lfp_and_spike_psds = np.zeros([len(pcf.df), nfreqs*2])
    nz = np.zeros(len(pcf.df), dtype='bool')
    for k,(lfp_index,spike_index) in enumerate(zip(pcf.df['lfp_index'], pcf.df['spike_index'])):
        lpsd = pcf.lfp_psds[lfp_index, :]
        spsd = pcf.spike_psds[spike_index, :]
        nz[k] = np.abs(lpsd).sum() > 0 and np.abs(spsd).sum() > 0
        lfp_and_spike_psds[k, :nfreqs] = lpsd
        lfp_and_spike_psds[k, nfreqs:] = spsd

    # zscore the concatenated matrix
    lfp_and_spike_psds = lfp_and_spike_psds[nz, :]
    lfp_and_spike_psds -= lfp_and_spike_psds.mean(axis=0)
    lfp_and_spike_psds /= lfp_and_spike_psds.std(axis=0, ddof=1)

    # compute the covariance
    lfp_and_spike_cov_est = LedoitWolf()
    lfp_and_spike_cov_est.fit(lfp_and_spike_psds)
    lfp_and_spike_cov = lfp_and_spike_cov_est.covariance_

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

    fig = plt.figure(figsize=(24, 12))
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.30, hspace=0.30)

    nrows = 2
    ncols = 100
    gs = plt.GridSpec(nrows, ncols)

    ax = plt.subplot(gs[0, :35])
    plt.errorbar(pcf.freqs, lfp_psd_mean, yerr=lfp_psd_std/np.sqrt(nsamps_lfp), c=COLOR_BLUE_LFP, linewidth=9.0, elinewidth=2.0, ecolor='k', alpha=0.7)
    plt.errorbar(pcf.freqs+3, spike_psd_mean, yerr=spike_psd_std/np.sqrt(nsamps_spike), c=COLOR_YELLOW_SPIKE, linewidth=9.0, elinewidth=2.0, ecolor='k', alpha=0.7)
    plt.legend(['LFP', 'Spike PSD'], fontsize='x-small')
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.ylim(0, 1)
    plt.title('Mean LFP and PSTH PSDs')

    fi = freqs < 200
    ax = plt.subplot(gs[1, :35])
    clrs = ['k', '#d60036', COLOR_YELLOW_SPIKE]
    alphas = [0.8, 0.8, 0.6]
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
        plt.plot(freqs[fi], psd_mean[fi], c=clrs[k], linewidth=9.0, alpha=alphas[k])
        plt.ylabel('Power (dB)')
        plt.xlabel('Frequency (Hz)')
        plt.axis('tight')
        plt.ylim(0, 1.0)
    plt.legend(['MLd', 'OV', 'Field L+CM'], fontsize='x-small', loc='upper right')
    plt.title('Mean PSTH PSDs (CRCNS Data)')

    ax = plt.subplot(gs[:, 40:])
    plt.imshow(lfp_and_spike_cov, aspect='auto', interpolation='nearest', origin='lower', cmap=magma, vmin=0, vmax=1)
    plt.colorbar()
    xy = np.arange(2*len(pcf.freqs))
    lbls = np.concatenate([['%d' % f for f in pcf.freqs], ['%d' % f for f in pcf.freqs]])
    plt.xticks(xy, lbls, rotation=45)
    plt.yticks(xy, lbls)
    plt.axhline(nfreqs-0.5, c='w')
    plt.axvline(nfreqs-0.5, c='w')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title('LFP-PSTH Correlation Matrix')

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