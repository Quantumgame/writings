import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from utils import get_freqs, set_font, get_this_dir
from zeebeez.aggregators.pairwise_decoders_multi_freq import AggregatePairwiseDecoder


def draw_figures():

    pfile = '/auto/tdrive/mschachter/data/aggregate/decoders_pairwise_coherence_multi_freq.h5'

    agg = AggregatePairwiseDecoder.load(pfile)

    nbands = agg.df['band'].max()
    sample_rate = 381.4697265625
    freqs = get_freqs(sample_rate)

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])

    """
    # TODO: compute the average likelihood ratio between intecept-only and full model for all sites!
    i = (agg.df['bird'] == 'GreBlu9508M') & (agg.df['block'] == 'Site4') & (agg.df['segment'] == 'Call1') & (agg.df['hemi'] == 'L') & (agg.df['band'] == 0)
    assert i.sum() == 1
    full_likelihood_for_null = agg.df['likelihood'][i].values[0]
    null_likelihood = 1.63  # for GreBlu9508_Site4_Call1_L
    null_likelihood_ratio = 2*(null_likelihood - full_likelihood_for_null)
    print 'full_likelihood_for_null=',full_likelihood_for_null
    print 'null_likelihood=',null_likelihood
    print 'null_likelihood_ratio=',null_likelihood_ratio
    """

    full_likelihoods = list()
    likelihood_ratios = list()
    pccs = list()
    pcc_thresh = 0.25
    single_band_likelihoods = list()
    single_band_pccs = list()
    for (bird,block,seg,hemi),gdf in g:
        # get the likelihood of the fullmodel
        i = gdf['band'] == 0
        assert i.sum() == 1

        num_samps = gdf[i]['num_samps'].values[0]
        print 'num_samps=%d' % num_samps
        full_likelihood = -gdf[i]['likelihood'].values[0] * num_samps
        pcc = gdf[i]['pcc'].values[0]

        if pcc < pcc_thresh:
            continue

        full_likelihoods.append(full_likelihood)
        pccs.append(pcc)

        # get the likelihood per frequency band
        ratio_by_band = np.zeros(nbands)
        single_likelihood_by_band = np.zeros(nbands)
        single_pcc_band = np.zeros(nbands)
        for k in range(nbands):
            i = (gdf['band'] == k+1) & (gdf['exfreq'] == True)
            assert i.sum() == 1

            num_samps2 = gdf[i]['num_samps'].values[0]
            assert num_samps2 == num_samps
            leftout_likelihood = -gdf[i]['likelihood'].values[0] * num_samps

            i = (gdf['band'] == k+1) & (gdf['exfreq'] == False)
            assert i.sum() == 1

            num_samps3 = gdf[i]['num_samps'].values[0]
            assert num_samps3 == num_samps2
            single_leftout_likelihood = -gdf[i]['likelihood'].values[0] * num_samps

            pcc = gdf[i]['pcc'].values[0]

            print '(%s,%s,%s,%s,%d) leftout=%0.6f, full=%0.6f, single=%0.6f, single_pcc=%0.6f, num_samps=%d' % \
                  (bird, block, seg, hemi, k, leftout_likelihood, full_likelihood, single_leftout_likelihood, pcc, num_samps)

            # compute the likelihood ratio
            lratio = -2*(leftout_likelihood - full_likelihood)
            ratio_by_band[k] = lratio
            single_likelihood_by_band[k] = single_leftout_likelihood
            single_pcc_band[k] = pcc

        likelihood_ratios.append(ratio_by_band)
        single_band_likelihoods.append(single_likelihood_by_band)
        single_band_pccs.append(single_pcc_band)

    pccs = np.array(pccs)
    likelihood_ratios = np.array(likelihood_ratios)
    full_likelihoods = np.array(full_likelihoods)
    single_band_likelihoods = np.array(single_band_likelihoods)
    single_band_pccs = np.array(single_band_pccs)

    # exclude segments whose likelihood ratio goes below zero
    # i = np.array([np.any(lrat < 0) for lrat in likelihood_ratios])
    i = np.ones(len(likelihood_ratios), dtype='bool')
    print 'i.sum()=%d' % i.sum()

    # compute significance threshold
    x = np.linspace(1, 150, 1000)
    df = 12
    p = chi2.pdf(x, df)
    sig_thresh = max(x[p > 0.01])

    # compute mean and std
    lrat_mean = likelihood_ratios[i, :].mean(axis=0)
    lrat_std = likelihood_ratios[i, :].std(axis=0, ddof=1)

    single_l_mean = single_band_likelihoods[i, :].mean(axis=0)
    single_l_std = single_band_likelihoods[i, :].std(axis=0, ddof=1)

    single_pcc_mean = single_band_pccs[i, :].mean(axis=0)
    single_pcc_std = single_band_pccs[i, :].std(axis=0, ddof=1)

    fig = plt.figure(figsize=(24, 16))
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    ax = plt.subplot(2, 3, 1)
    plt.plot(full_likelihoods, pccs, 'go', linewidth=2.0)
    plt.xlabel('log Likelihood')
    plt.ylabel('PCC')
    plt.axis('tight')

    ax = plt.subplot(2, 3, 2)
    for k,lrat in enumerate(likelihood_ratios[i, :]):
        plt.plot(freqs, lrat, '-', linewidth=2.0, alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Likelihood Ratio')
        plt.axis('tight')

    ax = plt.subplot(2, 3, 4)
    nsamps = len(likelihood_ratios)
    plt.errorbar(freqs, single_pcc_mean, yerr=single_pcc_std/np.sqrt(nsamps), ecolor='r', elinewidth=3.0, fmt='k-', linewidth=7.0, alpha=0.75)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PCC')
    plt.title('Mean Single Band Decoder PCC')
    plt.axis('tight')

    ax = plt.subplot(2, 3, 5)
    plt.errorbar(freqs, single_l_mean, yerr=single_l_std/np.sqrt(nsamps), ecolor='r', elinewidth=3.0, fmt='k-', linewidth=7.0, alpha=0.75)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('log Likelihood')
    plt.title('Mean Single Band Decoder Likelihood')
    plt.axis('tight')

    ax = plt.subplot(2, 3, 6)
    plt.errorbar(freqs, lrat_mean, yerr=lrat_std/np.sqrt(nsamps), ecolor='r', elinewidth=3.0, fmt='k-', linewidth=7.0, alpha=0.75)
    plt.plot(freqs, np.ones_like(freqs)*sig_thresh, 'k--', linewidth=7.0, alpha=0.75)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Likelihood Ratio')
    plt.title('Mean Likelihood Ratio')
    plt.axis('tight')
    plt.ylim(0, lrat_mean.max())



    fname = os.path.join(get_this_dir(), 'figs.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


if __name__ == '__main__':
    set_font()
    draw_figures()