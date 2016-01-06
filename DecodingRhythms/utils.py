import os
import sys

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from lasp.signal import coherency
from lasp.timefreq import power_spectrum_jn
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


COLOR_BLUE_LFP = '#0068A5'
COLOR_YELLOW_SPIKE = '#F0DB00'
COLOR_RED_SPIKE_RATE = '#E90027'

def set_font():
    font = {'family':'normal', 'weight':'bold', 'size':16}
    matplotlib.rc('font', **font)


def get_this_dir():
    """ Get the directory that contains the python file that is calling this function. """

    f = sys._current_frames().values()[0]
    calling_file_path = f.f_back.f_globals['__file__']
    root_dir,fname = os.path.split(calling_file_path)
    return root_dir


def get_freqs(sample_rate, window_length=0.060, increment=None):
    if increment is None:
        increment = 2.0 / sample_rate
    nt = int(window_length*2*sample_rate)
    s = np.random.randn(nt)
    pfreq,psd1,ps_var,phase = power_spectrum_jn(s, sample_rate, window_length, increment)
    return pfreq


def get_lags_ms(sample_rate, lags=np.arange(-20, 21, 1)):
    return (lags / sample_rate)*1e3


def log_transform(s):

    nz = s > 0
    s[nz] = 20*np.log10(s[nz]) + 70
    s[s < 0] = 0


def compute_spectra_and_coherence_single_electrode(lfp1, lfp2, sample_rate, e1, e2,
                                                   window_length=0.060, increment=None, log=True,
                                                   window_fraction=0.60, noise_floor_db=25,
                                                   lags=np.arange(-20, 21, 1), psd_stats=None):
    """

    :param lfp1: An array of shape (ntrials, nt)
    :param lfp2: An array of shape (ntrials, nt)
    :return:
    """

    # compute the mean (locked) spectra
    lfp1_mean = lfp1.mean(axis=0)
    lfp2_mean = lfp2.mean(axis=0)

    if increment is None:
        increment = 2.0 / sample_rate

    pfreq,psd1,ps_var,phase = power_spectrum_jn(lfp1_mean, sample_rate, window_length, increment)
    pfreq,psd2,ps_var,phase = power_spectrum_jn(lfp2_mean, sample_rate, window_length, increment)
    
    if log:
        log_transform(psd1)
        log_transform(psd2)

    c12 = coherency(lfp1_mean, lfp2_mean, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)

    # compute the nonlocked spectra coherence
    c12_pertrial = list()
    ntrials,nt = lfp1.shape
    psd1_ms_all = list()
    psd2_ms_all = list()
    for k in range(ntrials):
        i = np.ones([ntrials], dtype='bool')
        i[k] = False
        lfp1_jn_mean = lfp1[i, :].mean(axis=0)
        lfp2_jn_mean = lfp2[i, :].mean(axis=0)

        lfp1_ms = lfp1[k, :] - lfp1_jn_mean
        lfp2_ms = lfp2[k, :] - lfp2_jn_mean

        pfreq,psd1_ms,ps_var_ms,phase_ms = power_spectrum_jn(lfp1_ms, sample_rate, window_length, increment)
        pfreq,psd2_ms,ps_var_ms,phase_ms = power_spectrum_jn(lfp2_ms, sample_rate, window_length, increment)
        if log:
            log_transform(psd1_ms)
            log_transform(psd2_ms)

        psd1_ms_all.append(psd1_ms)
        psd2_ms_all.append(psd2_ms)

        c12_ms = coherency(lfp1_ms, lfp2_ms, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)
        c12_pertrial.append(c12_ms)

    psd1_ms_all = np.array(psd1_ms_all)
    psd2_ms_all = np.array(psd2_ms_all)
    psd1_ms = psd1_ms_all.mean(axis=0)
    psd2_ms = psd2_ms_all.mean(axis=0)

    if psd_stats is not None:
        psd_mean1,psd_std1 = psd_stats[e1]
        psd_mean2,psd_std2 = psd_stats[e2]
        psd1 -= psd_mean1
        psd1 /= psd_std1
        psd2 -= psd_mean2
        psd2 /= psd_std2

        psd1_ms -= psd_mean1
        psd1_ms /= psd_std1
        psd2_ms -= psd_mean2
        psd2_ms /= psd_std2

    c12_pertrial = np.array(c12_pertrial)
    c12_nonlocked = c12_pertrial.mean(axis=0)

    # compute the coherence per trial then take the average
    c12_totals = list()
    for k in range(ntrials):
        c12 = coherency(lfp1[k, :], lfp2[k, :], lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)
        c12_totals.append(c12)

    c12_totals = np.array(c12_totals)
    c12_total = c12_totals.mean(axis=0)

    return pfreq, psd1, psd2, psd1_ms, psd2_ms, c12, c12_nonlocked, c12_total


def compute_spectra_and_coherence_multi_electrode_single_trial(lfps, sample_rate, electrode_indices, electrode_order,
                                                               window_length=0.060, increment=None, log=True,
                                                               window_fraction=0.60, noise_floor_db=25,
                                                               lags=np.arange(-20, 21, 1),
                                                               psd_stats=None):
    """
    :param lfps: an array of shape (ntrials, nelectrodes, nt)
    :return:
    """

    if increment is None:
        increment = 2.0 / sample_rate

    nelectrodes,nt = lfps.shape
    freqs = get_freqs(sample_rate, window_length, increment)
    lags_ms = get_lags_ms(sample_rate, lags)

    spectra = np.zeros([nelectrodes, len(freqs)])
    cross_mat = np.zeros([nelectrodes, nelectrodes, len(lags_ms)])

    for k in range(nelectrodes):

        _e1 = electrode_indices[k]
        i1 = electrode_order.index(_e1)

        lfp1 = lfps[k, :]

        freqs,psd1,ps_var,phase = power_spectrum_jn(lfp1, sample_rate, window_length, increment)
        if log:
            log_transform(psd1)

        if psd_stats is not None:
            psd_mean,psd_std = psd_stats[_e1]

            """
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.plot(freqs, psd1, 'k-')
            plt.title('PSD (%d)' % _e1)
            plt.axis('tight')

            plt.subplot(2, 2, 3)
            plt.plot(freqs, psd_mean, 'g-')
            plt.title('Mean')
            plt.axis('tight')

            plt.subplot(2, 2, 4)
            plt.plot(freqs, psd_std, 'c-')
            plt.title('STD')
            plt.axis('tight')

            plt.subplot(2, 2, 2)
            psd1_z = deepcopy(psd1)
            psd1_z -= psd_mean
            psd1_z /= psd_std
            plt.plot(freqs, psd1_z, 'r-')
            plt.title('Zscored')
            plt.axis('tight')
            """
            psd1 -= psd_mean
            psd1 /= psd_std

        spectra[i1, :] = psd1

        for j in range(k):

            _e2 = electrode_indices[j]
            i2 = electrode_order.index(_e2)

            lfp2 = lfps[j, :]

            cf = coherency(lfp1, lfp2, lags, window_fraction=window_fraction, noise_floor_db=noise_floor_db)

            """
            freqs,c12,c_var_amp,c_phase,c_phase_var,coherency,coherency_t = coherence_jn(lfp1, lfp2, sample_rate,
                                                                                         window_length, increment,
                                                                                         return_coherency=True)
            """

            cross_mat[i1, i2] = cf
            cross_mat[i2, i1] = cf[::-1]

    return spectra, cross_mat


def add_region_info(agg, df):
    """ Make a new DataFrame that contains region information. """

    edf = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/electrode_data.csv')

    new_data = dict()
    for key in df.keys():
        new_data[key] = list()

    # peek into the aggregate data to get a list of class names
    k1 = agg.class_names.keys()[0]
    stim_class_names = agg.class_names[k1][0]

    new_data['reg1'] = list()
    new_data['reg2'] = list()
    new_data['gs'] = list()  # global selectivity

    for cname in stim_class_names:
        new_data['pcc_%s' % cname] = list()
        new_data['sel_%s' % cname] = list()

    # make a map of bird/block/hemi/electrode for fast lookup
    emap = dict()
    for k,row in edf.iterrows():
        key = (row['bird'], row['block'], row['hemisphere'], row['electrode'])
        reg = row['region']
        emap[key] = reg

    for k,row in df.iterrows():
        for key in df.keys():
            if key == 'segment' and row[key] == 'Call1c':
                new_data[key].append('Call1')
            else:
                new_data[key].append(row[key])

        bird = row['bird']
        block = row['block']
        hemi = row['hemi']
        e1 = row['e1']
        e2 = row['e2']

        # get the confusion matrix for this row
        index = row['index']
        mat_key = (row['decomp'], row['order'], row['ptype'])
        C = agg.confidence_matrices[mat_key][index]
        cnames = agg.class_names[mat_key][index]

        # compute the pcc fraction for each category
        pcc_fracs = np.zeros([len(cnames)])
        for k,cname in enumerate(cnames):
            p = C[k]
            p /= p.sum()
            pcc_fracs[k] = p[k]
            new_data['pcc_%s' % cname].append(p[k])

        # compute the selectivity for each category
        for k,cname in enumerate(cnames):
            i = np.ones(len(cnames), dtype='bool')
            i[k] = False
            sel = np.log2(((len(cnames)-1)*pcc_fracs[k]) / pcc_fracs[i].sum())
            new_data['sel_%s' % cname].append(sel)

        # normalize the fractions so they become a distribution
        pcc_fracs /= pcc_fracs.sum()

        # compute the global selectivity
        if np.isnan(pcc_fracs).sum() > 0:
            gs = 0
        else:
            nz = pcc_fracs > 0
            assert np.abs(pcc_fracs.sum() - 1) < 1e-6, "pcc_fracs.sum()=%f" % pcc_fracs.sum()
            Hobs = -np.sum(pcc_fracs[nz]*np.log2(pcc_fracs[nz]))
            Hmax = np.log2(len(cnames))
            gs = 1. - (Hobs / Hmax)
        new_data['gs'].append(gs)

        key = (bird, block, hemi, e1)
        reg1 = emap[key]

        key = (bird, block, hemi, e2)
        reg2 = emap[key]

        reg1 = reg1.replace('L2b', 'L2')
        reg1 = reg1.replace('L2A', 'L2')
        reg1 = reg1.replace('L2B', 'L2')

        reg2 = reg2.replace('L2b', 'L2')
        reg2 = reg2.replace('L2A', 'L2')
        reg2 = reg2.replace('L2B', 'L2')

        new_data['reg1'].append(reg1)
        new_data['reg2'].append(reg2)

    return pd.DataFrame(new_data),stim_class_names


def get_psd_stats(bird, block, seg, hemi, data_dir='/auto/tdrive/mschachter/data'):

    transforms_dir = os.path.join(data_dir, bird, 'transforms')
    cf_file = os.path.join(transforms_dir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird, block, seg, hemi))
    cft = PairwiseCFTransform.load(cf_file)

    electrodes = cft.df.electrode1.unique()

    estats = dict()
    for e in electrodes:
        i = (cft.df.electrode1 == e) & (cft.df.electrode1 == cft.df.electrode2) & (cft.df.decomp == 'locked')
        indices = cft.df['index'][i].values
        psds = cft.psds[indices]
        log_transform(psds)
        estats[e] = (psds.mean(axis=0), psds.std(axis=0, ddof=1))
    return estats


def compute_avg_and_ms(lfp):

    lfp_mean = lfp.mean(axis=0)

    lfp_ms_all = list()
    ntrials,nelectrodes = lfp.shape
    for k in range(ntrials):
        i = np.ones([ntrials], dtype='bool')
        i[k] = False
        lfp_resid = lfp[k, :] - lfp[i].mean(axis=0)
        lfp_ms_all.append(lfp_resid)

    lfp_ms_all = np.array(lfp_ms_all)
    lfp_ms = lfp_ms_all.mean(axis=0)

    return lfp_mean, lfp_ms


if __name__ == '__main__':
    sample_rate = 381.4697265625
    print 'freqs=',get_freqs(sample_rate)


def clean_region(reg):
    if '-' in reg:
        return '?'
    if reg.startswith('L2'):
        return 'L2'
    if reg == 'CM':
        return '?'
    return reg