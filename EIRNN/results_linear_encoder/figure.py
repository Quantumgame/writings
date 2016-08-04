import os
import operator

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.mixture import GMM

from DecodingRhythms.utils import set_font, get_this_dir, log_transform
from lasp.plots import boxplot_with_colors
from lasp.timefreq import power_spectrum_jn
from zeebeez.aggregators.lfp_encoder import AggregateLFPEncoder


def draw_perfs(agg):

    # filter out some data points
    i = (agg.df.cc > 0.05) & (agg.df.region != '?') & (agg.df.region != 'HP') \
        & ~np.isnan(agg.df.dist_l2a) & ~np.isnan(agg.df.dist_midline) \
        & (agg.df.dist_l2a > -1) & (agg.df.dist_l2a < 1)
    df = agg.df[i]

    print len(df)
    print df.region.unique()

    # make a scatter plot of performance by anatomical location
    fig = plt.figure(figsize=(23, 10), facecolor='w')
    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.25)

    gs = plt.GridSpec(2, 100)

    ax = plt.subplot(gs[:, :55])
    plt.scatter(df.dist_midline, df.dist_l2a, c=df.cc, s=64, cmap=plt.cm.afmhot_r, alpha=0.7)
    plt.ylim(-1, 1)
    plt.xlim(0, 2.5)
    for x, y, r in zip(df.dist_midline, df.dist_l2a, df.region):
        plt.text(x, y - 0.05, r, fontsize=9)
    cbar = plt.colorbar(label='Linear Encoder Performance (cc)')
    plt.xlabel('Dist to Midline (mm)')
    plt.ylabel('Dist to L2A (mm)')

    region_stats = list()
    regions_to_use = ['L1', 'L2', 'L3', 'CM', 'NCM']
    for reg in regions_to_use:
        i = df.region == reg
        region_stats.append({'cc': df[i].cc, 'cc_mean': df[i].cc.mean(), 'region': reg})
    region_stats.sort(key=operator.itemgetter('cc_mean'), reverse=True)
    regions = [x['region'] for x in region_stats]
    region_ccs = {x['region']: x['cc'] for x in region_stats}

    ax = plt.subplot(gs[:45, 65:])
    boxplot_with_colors(region_ccs, group_names=regions, ax=ax, group_colors=['k'] * len(regions), box_alpha=0.7)
    plt.xlabel('Region')
    plt.ylabel('Linear Encoder Performance (cc)')

    fname = os.path.join(get_this_dir(), 'encoder_perf.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')


def compute_best_freq(filts, sample_rate, lags_ms):

    def sine_func(_t, _f, _phase):
        return -np.sin(2*np.pi*_f*_t + _phase)

    best_freqs = list()
    fi = (lags_ms > 5) & (lags_ms < 80.)
    t = lags_ms[fi] * 1e-3
    t -= t.min()
    for f in filts:
        filt_norm = f[fi]
        filt_norm /= np.abs(filt_norm).max()

        try:
            popt, pcov = curve_fit(sine_func, t, filt_norm, p0=[20, 0])
        except RuntimeError:
            best_freqs.append(-1)
            continue

        ypred = sine_func(t, *popt)
        bfreq = np.abs(popt[0])
        best_freqs.append(bfreq)

        if bfreq < 10:
            plt.figure()
            plt.plot(t, filt_norm, 'k-')
            plt.plot(t, ypred, 'r-', alpha=0.7)
            plt.title('f=%0.3fHz' % popt[0])
            plt.axis('tight')
            plt.show()

    return np.array(best_freqs)


def compute_filt_power_spectra(filts, sample_rate, lags_ms):

    filt_freq = None
    psds = list()
    # fi = (lags_ms > 8.) & (lags_ms < 125.)
    # fi = (lags_ms > 8.)
    fi = (lags_ms > -1)
    for f in filts:
        filt_freq,filt_ps,filt_ps_var,filt_phase = power_spectrum_jn(f[fi], sample_rate, 0.250, 0.025)
        psds.append(filt_ps)

    psds = np.array(psds)
    psds /= psds.max()
    # log_transform(psds)
    fi = filt_freq < 40.
    return filt_freq[fi], psds[:, fi]


def draw_filters(agg):

    # filter out some data points
    i = (agg.df.cc > 0.20) & (agg.df.region != '?') & (agg.df.region != 'HP')
    df = agg.df[i]

    regions_to_use = ['L1', 'L2', 'L3', 'CM', 'NCM']
    region_filters = list()
    lags_ms = (agg.lags/agg.sample_rate)*1e3
    for reg in regions_to_use:
        i = df.region == reg
        xi = df.xindex[i].values
        filts = agg.filters[xi, :]
        # quantify peak time
        filt_peaks = compute_filt_peaks(filts, lags_ms)

        #quantify center frequency
        center_freqs = compute_best_freq(filts, agg.sample_rate, lags_ms)

        # compute power spectra
        # filt_ps_freq,filt_psds = compute_filt_power_spectra(filts, agg.sample_rate, lags_ms)

        region_filters.append({'filters':filts, 'region':reg, 'peak_mean':filt_peaks.mean(), 'freqs':center_freqs})

    region_filters.sort(key=operator.itemgetter('peak_mean'))

    topn = 20
    lag_i = (lags_ms < 100.)
    fig = plt.figure(figsize=(15, 13), facecolor='w')
    fig.subplots_adjust(hspace=0.35, wspace=0.35, top=0.95, bottom=0.05)

    gs = plt.GridSpec(len(region_filters), 2)

    nrows = len(region_filters)
    for k,rdict in enumerate(region_filters):

        ax = plt.subplot(gs[k, 0])
        plt.axhline(0, c='k')
        for f in rdict['filters'][:topn]:
            plt.plot(lags_ms[lag_i], f[lag_i]*1e3, 'k-', alpha=0.7, linewidth=2.0)
        plt.axis('tight')
        plt.ylim(-4, 4)
        plt.title(rdict['region'])
        if k == len(region_filters)-1:
            plt.xlabel('Filter Lag (ms)')

        ax = plt.subplot(gs[k, 1])
        cfreqs = rdict['freqs']
        i = (cfreqs > 0)
        plt.hist(cfreqs[i], bins=20, color='b', alpha=0.7, normed=False, range=(10, 30))
        plt.title(rdict['region'])
        if k == len(region_filters) - 1:
            plt.xlabel('Filter Frequency (Hz)')
        plt.ylabel('Count')
        plt.axis('tight')
        plt.xlim(10, 30)

    fname = os.path.join(get_this_dir(), 'encoder_filters.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()


def compute_filt_peaks(filts, lags_ms):

    peaks = np.zeros(filts.shape[0])
    for k,f in enumerate(filts):
        npeak = np.argmin(f)
        peaks[k] = lags_ms[npeak]
    return peaks


def stats(agg):

    lags_ms = (agg.lags / agg.sample_rate) * 1e3

    i = (agg.df.cc > 0.20)
    xi = agg.df[i].xindex.values
    filts = agg.filters[xi, :]

    cfreqs = compute_best_freq(filts, agg.sample_rate, lags_ms)
    cfreqs = cfreqs[cfreqs > 0]
    cfreqs = cfreqs.reshape([len(cfreqs), 1])

    gmm1 = GMM(n_components=1)
    gmm1.fit(cfreqs)
    lk_null = gmm1.score(cfreqs).sum()
    aic_null = gmm1.aic(cfreqs)

    gmm2 = GMM(n_components=2)
    gmm2.fit(cfreqs)
    print 'Center frequencies of 2-component GMM:',gmm2.means_.squeeze()
    print 'Covariances: ',np.sqrt(gmm2.covars_.squeeze())
    lk_full = gmm2.score(cfreqs).sum()
    aic_full = gmm2.aic(cfreqs)

    lk_rat = -2*(lk_null - lk_full)

    print 'Null likelihood: %0.6f' % lk_null
    print 'Full likelihood: %0.6f' % lk_full
    print 'Likelihood Ratio: %0.6f' % lk_rat

    print 'Null AIC: %0.6f' % aic_null
    print 'Full AIC: %0.6f' % aic_full


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_encoder.h5')
    agg = AggregateLFPEncoder.load(agg_file)

    agg.df.to_csv('/auto/tdrive/mschachter/data/aggregate/lfp_encoder.csv', header=True, index=False)

    # draw_perfs(agg)

    # draw_filters(agg)

    stats(agg)

    plt.show()


if __name__ == '__main__':

    set_font()
    draw_figures()


