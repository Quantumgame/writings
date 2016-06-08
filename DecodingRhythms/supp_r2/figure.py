import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zeebeez.aggregators.acoustic_encoder_decoder import AcousticEncoderDecoderAggregator


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'pard.h5')
    agg = AcousticEncoderDecoderAggregator.load(agg_file)

    decomps = ['spike_rate', 'full_psds', 'spike_rate+spike_sync', 'full_psds+full_cfs']

    r2_vals = dict()
    for decomp in decomps:
        i = agg.df.decomp == decomp
        assert i.sum() > 0

        enc_r2 = list()
        enc_cv_r2 = list()
        dec_r2 = list()
        dec_cv_r2 = list()
        for wkey in agg.df.wkey[i].values:
            eperf = agg.encoder_perfs[wkey].ravel()
            eperf_cv = agg.encoder_cv_perfs[wkey].ravel()
            enc_r2.extend(eperf)
            enc_cv_r2.extend(eperf_cv)

            dperf = agg.decoder_perfs[wkey].ravel()
            dperf_cv = agg.decoder_cv_perfs[wkey].ravel()
            dec_r2.extend(dperf)
            dec_cv_r2.extend(dperf_cv)

        print '############ %s ###########' % decomp
        for r2,cvr2 in zip(dec_r2, dec_cv_r2):
            print '%0.6f, %0.6f' % (r2, cvr2)

        r2_vals[decomp] = {'enc_r2':np.array(enc_r2), 'enc_cv_r2':np.array(enc_cv_r2),
                           'dec_r2': np.array(dec_r2), 'dec_cv_r2': np.array(dec_cv_r2),}

    figsize = (23, 10)
    plt.figure(figsize=figsize)

    nrows = 2
    ncols = len(decomps)
    gs = plt.GridSpec(nrows, ncols)
    for k,decomp in enumerate(decomps):

        ax = plt.subplot(gs[0, k])
        x = np.linspace(0, 1, 50)
        plt.plot(x, x, 'k-', alpha=0.5)
        plt.plot(r2_vals[decomp]['enc_cv_r2'], r2_vals[decomp]['enc_r2'], 'ko', alpha=0.7)
        plt.xlabel('Generalization R2')
        plt.ylabel('Full R2')
        plt.axis('tight')
        plt.title('Encoder: %s' % decomp)

        ax = plt.subplot(gs[1, k])
        x = np.linspace(0, 1, 50)
        plt.plot(x, x, 'k-', alpha=0.5)
        plt.plot(r2_vals[decomp]['dec_cv_r2'], r2_vals[decomp]['dec_r2'], 'ko', alpha=0.7)
        plt.xlabel('Generalization R2')
        plt.ylabel('Full R2')
        plt.axis('tight')
        plt.title('Decoder: %s' % decomp)

    plt.show()


if __name__ == '__main__':
    draw_figures()
