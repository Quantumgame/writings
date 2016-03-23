import os
from copy import deepcopy

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from lasp.plots import custom_legend, multi_plot

from lasp.colormaps import magma
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

from DecodingRhythms.utils import COLOR_YELLOW_SPIKE, get_this_dir, COLOR_BLUE_LFP
from lasp.sound import log_transform
from utils import set_font
from zeebeez.aggregators.pairwise_cf import AggregatePairwiseCF
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def plot_psds(psd_file, data_dir='/auto/tdrive/mschachter/data'):

    # read PairwiseCF file
    pcf_file = os.path.join(data_dir, 'aggregate', 'pairwise_cf.h5')
    pcf = AggregatePairwiseCF.load(pcf_file)
    # pcf.zscore_within_site()

    g = pcf.df.groupby(['bird', 'block', 'segment', 'electrode'])
    nsamps_electrodes = len(g)

    i = pcf.df.cell_index != -1
    g = pcf.df[i].groupby(['bird', 'block', 'segment', 'electrode', 'cell_index'])
    nsamps_cells = len(g)

    print '# of electrodes: %d' % nsamps_electrodes
    print '# of cells: %d' % nsamps_cells
    print '# of lfp samples: %d' % (pcf.lfp_psds.shape[0])
    print '# of spike psd samples: %d' % (pcf.spike_psds.shape[0])

    # compute the LFP mean and std
    lfp_psds = deepcopy(pcf.lfp_psds)
    print 'lfp_psds_ind: max=%f, q99=%f' % (lfp_psds.max(), np.percentile(lfp_psds.ravel(), 99))
    log_transform(lfp_psds)
    print 'lfp_psds_ind: max=%f, q99=%f' % (lfp_psds.max(), np.percentile(lfp_psds.ravel(), 99))
    nz = lfp_psds.sum(axis=1) > 0
    lfp_psds = lfp_psds[nz, :]
    lfp_psd_mean = lfp_psds.mean(axis=0)
    lfp_psd_std = lfp_psds.std(axis=0, ddof=1)
    nsamps_lfp = lfp_psds.shape[0]

    # get the spike rate
    spike_rate = pcf.df.spike_rate.values
    # plt.figure()
    # plt.hist(spike_rate, bins=20, color='g', alpha=0.7)
    # plt.title('Spike Rate Histogram, q1=%0.3f, q5=%0.3f, q10=%0.3f, q50=%0.3f, q99=%0.3f' %
    #           (np.percentile(spike_rate, 1), np.percentile(spike_rate, 5), np.percentile(spike_rate, 10),
    #           np.percentile(spike_rate, 50), np.percentile(spike_rate, 99)))
    # plt.show()

    # compute the covariance
    lfp_psd_z = deepcopy(lfp_psds)
    lfp_psd_z -= lfp_psd_mean
    lfp_psd_z /= lfp_psd_std
    lfp_and_spike_cov_est = LedoitWolf()
    lfp_and_spike_cov_est.fit(lfp_psd_z)
    lfp_and_spike_cov = lfp_and_spike_cov_est.covariance_

    """
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
    """

    # resample the lfp mean and std
    freq_rs = np.linspace(pcf.freqs.min(), pcf.freqs.max(), 1000)
    
    lfp_mean_cs = interp1d(pcf.freqs, lfp_psd_mean, kind='cubic')
    lfp_mean_rs = lfp_mean_cs(freq_rs)
    
    lfp_std_cs = interp1d(pcf.freqs, lfp_psd_std, kind='cubic')
    lfp_std_rs = lfp_std_cs(freq_rs)

    # concatenate the lfp psd and log spike rate
    lfp_psd_and_spike_rate = list()
    for k,(li,si) in enumerate(zip(pcf.df['lfp_index'], pcf.df['spike_index'])):
        lpsd = pcf.lfp_psds[li, :]
        srate,sstd = pcf.spike_rates[si, :]
        if srate > 0:
            lfp_psd_and_spike_rate.append(np.hstack([lpsd, np.log(srate)]))
    lfp_psd_and_spike_rate = np.array(lfp_psd_and_spike_rate)

    nfreqs = len(pcf.freqs)
    lfp_rate_cc = np.zeros([nfreqs])
    for k in range(nfreqs):
        lfp_rate_cc[k] = np.corrcoef(lfp_psd_and_spike_rate[:, k], lfp_psd_and_spike_rate[:, -1])[0, 1]

    fig = plt.figure(figsize=(24, 12))
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.30, hspace=0.30)

    nrows = 2
    ncols = 100
    gs = plt.GridSpec(nrows, ncols)

    ax = plt.subplot(gs[0, :35])
    plt.errorbar(freq_rs, lfp_mean_rs, yerr=lfp_std_rs, c='k', linewidth=9.0, elinewidth=3.0,
                 ecolor='#D8D8D8', alpha=0.5, capthick=0.)
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    # plt.ylim(0, 1)
    plt.title('Mean LFP PSD')

    ax = plt.subplot(gs[1, :35])
    plt.plot(pcf.freqs, lfp_rate_cc, '-', c=COLOR_BLUE_LFP, linewidth=9.0, alpha=0.7)
    plt.axhline(0, c='k')
    plt.axis('tight')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Correlation Coefficient')
    plt.ylim(-0.05, 0.25)
    plt.title('LFP Power vs log Spike Rate')

    """
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
    """

    ax = plt.subplot(gs[:, 40:])
    plt.imshow(lfp_and_spike_cov, aspect='auto', interpolation='nearest', origin='lower', cmap=magma, vmin=0, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    xy = np.arange(len(pcf.freqs))
    lbls = ['%d' % f for f in pcf.freqs]
    plt.xticks(xy, lbls, rotation=0)
    plt.yticks(xy, lbls)
    plt.axhline(nfreqs-0.5, c='w')
    plt.axvline(nfreqs-0.5, c='w')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.title('LFP PSD Correlation Matrix')

    fname = os.path.join(get_this_dir(), 'crcns_data.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_spike_rate_vs_power(data_dir='/auto/tdrive/mschachter/data'):

    # read PairwiseCF file
    pcf_file = os.path.join(data_dir, 'aggregate', 'pairwise_cf.h5')
    pcf = AggregatePairwiseCF.load(pcf_file)

    # concatenate the lfp and spike psds
    nfreqs = len(pcf.freqs)
    lfp_and_spike_psds = np.zeros([len(pcf.df), nfreqs*2 + 1])
    nz = np.zeros(len(pcf.df), dtype='bool')
    for k,(lfp_index,spike_index) in enumerate(zip(pcf.df['lfp_index'], pcf.df['spike_index'])):
        lpsd = pcf.lfp_psds[lfp_index, :]
        spsd = pcf.spike_psds[spike_index, :]
        srate,sstd = pcf.spike_rates[spike_index, :]
        nz[k] = np.abs(lpsd).sum() > 0 and np.abs(spsd).sum() > 0
        lfp_and_spike_psds[k, :nfreqs] = lpsd
        lfp_and_spike_psds[k, nfreqs:-1] = spsd
        lfp_and_spike_psds[k, -1] = np.log(srate)

    # throw some bad data points out
    lfp_sum = lfp_and_spike_psds[:, :nfreqs].sum(axis=1)
    spike_sum =  lfp_and_spike_psds[:, nfreqs:-1].sum(axis=1)
    nz = ~np.isinf(lfp_and_spike_psds[:, -1]) & (lfp_sum > 0) & (spike_sum > 0) & ~np.isnan(spike_sum) & ~np.isnan(lfp_sum)
    print '# of good data points: %d out of %d' % (nz.sum(), lfp_and_spike_psds.shape[0])

    # zscore the concatenated matrix
    lfp_and_spike_psds = lfp_and_spike_psds[nz, :]
    lfp_and_spike_psds -= lfp_and_spike_psds.mean(axis=0)
    lfp_and_spike_psds /= lfp_and_spike_psds.std(axis=0, ddof=1)

    # compute CC between spike rate and power
    lfp_spike_rate_cc = np.zeros(len(pcf.freqs))
    spike_spike_rate_cc = np.zeros(len(pcf.freqs))

    for k,f in enumerate(pcf.freqs):
        lfp_spike_rate_cc[k] = np.corrcoef(lfp_and_spike_psds[:, k], lfp_and_spike_psds[:, -1])[0, 1]
        spike_spike_rate_cc[k] = np.corrcoef(lfp_and_spike_psds[:, k+len(pcf.freqs)], lfp_and_spike_psds[:, -1])[0, 1]

    fig = plt.figure(figsize=(12, 7))
    plt.axhline(0, c='k')
    plt.plot(pcf.freqs, lfp_spike_rate_cc, '-', linewidth=7.0, alpha=0.7, c=COLOR_BLUE_LFP)
    plt.plot(pcf.freqs, spike_spike_rate_cc, '-', linewidth=7.0, alpha=0.7, c=COLOR_YELLOW_SPIKE)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Correlation Coefficient')
    plt.title('CC Between Log Spike Rate and Spectral Power')
    plt.axis('tight')
    plt.ylim(-0.1, 0.6)
    leg = custom_legend([COLOR_BLUE_LFP, COLOR_YELLOW_SPIKE], ['LFP PSD', 'Spike PSD'])
    plt.legend(handles=leg, fontsize='x-small')

    fname = os.path.join(get_this_dir(), 'power_vs_rate.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures():
    crcns_dir='/auto/tdrive/mschachter/data/crcns'
    psd_file = os.path.join(crcns_dir, 'cell_psd.h5')
    plot_psds(psd_file)
    # draw_spike_rate_vs_power()

    plt.show()


if __name__ == '__main__':
    set_font()
    draw_figures()