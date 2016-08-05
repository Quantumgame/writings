import os
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import set_font, get_this_dir, clean_region
from lasp.plots import custom_legend
from zeebeez.aggregators.rnn import RNNAggregator
from zeebeez.transforms.rnn_preprocess import RNNPreprocessTransform, visualize_pred_file
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT


def stats(agg, data_dir='/auto/tdrive/mschachter/data'):
    data = {'bird': list(), 'block': list(), 'segment': list(), 'hemi': list(), 'electrode': list(),
            'linear_cc': list(), 'cc': list(), 'err': list(),
            'lambda1': list(), 'lambda2': list(), 'n_unit': list(), 'region':list(), 'md5':list()}

    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data+dist.csv'))

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird, block, segment, hemi), gdf in g:

        perfs = list()
        gg = gdf.groupby(['lambda1', 'lambda2', 'n_unit'])
        for (lambda1, lambda2, n_unit), ggdf in gg:
            err = ggdf.err.values[0]
            perfs.append({'err': err, 'lambda1': lambda1, 'lambda2': lambda2, 'n_unit': n_unit})

        perfs.sort(key=operator.itemgetter('err'))

        best_lambda1 = perfs[0]['lambda1']
        best_lambda2 = perfs[0]['lambda2']
        best_n_unit = perfs[0]['n_unit']
        best_err = perfs[0]['err']

        print 'err=%0.3f, lambda1=%0.3f, lambda2=%0.3f, n_unit=%d' % (best_err, best_lambda1, best_lambda2, best_n_unit)

        i = (gdf.lambda1 == best_lambda1) & (gdf.lambda2 == best_lambda2) & (gdf.n_unit == best_n_unit)
        assert i.sum() == 16, 'i.sum()=%d' % i.sum()

        for e in gdf[i].electrode.unique():
            ii = (gdf.lambda1 == best_lambda1) & (gdf.lambda2 == best_lambda2) & (gdf.n_unit == best_n_unit) & (
            gdf.electrode == e)
            assert ii.sum() == 1, 'ii.sum()=%d' % ii.sum()

            iii = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
            assert iii.sum() == 1, 'iii.sum()=%d' % iii.sum()
            reg = clean_region(edata[iii].region.values[0])

            data['bird'].append(bird)
            data['block'].append(block)
            data['segment'].append(segment)
            data['hemi'].append(hemi)
            data['lambda1'].append(best_lambda1)
            data['lambda2'].append(best_lambda2)
            data['n_unit'].append(best_n_unit)
            data['err'].append(best_err)
            data['electrode'].append(e)
            data['linear_cc'].append(gdf[ii].linear_cc.values[0])
            data['cc'].append(gdf[ii].cc.values[0])
            data['region'].append(reg)
            data['md5'].append(gdf[ii].md5.values[0])

    df = pd.DataFrame(data)
    df.to_csv('/auto/tdrive/mschachter/data/aggregate/rnn_best.csv', header=True, index=False)

    fig = plt.figure(figsize=(12, 10), facecolor='w')
    x = np.linspace(0, 1, 20)
    plt.plot(x, x, 'k-')
    plt.plot(df.linear_cc, df.cc, 'go', alpha=0.7, markersize=12)
    plt.xlabel('Linear CC')
    plt.ylabel('RNN CC')
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.8)

    fname = os.path.join(get_this_dir(), 'linear_vs_rnn_cc.svg')
    # plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()


def draw_preds(agg, data_dir='/auto/tdrive/mschachter/data'):

    df_best = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/rnn_best.csv')

    # print 'min_err=', df_best.err.min()

    bird = 'GreBlu9508M'
    block = 'Site2'
    segment = 'Call2'
    hemi = 'L'
    best_md5 = 'db5d601c8af2621f341d8e4cbd4108c1'

    fname = '%s_%s_%s_%s' % (bird, block, segment, hemi)
    preproc_file = os.path.join(data_dir, bird, 'preprocess', 'RNNPreprocess_%s.h5' % fname)
    rnn_file = os.path.join(data_dir, bird, 'rnn', 'RNNLFPEncoder_%s_%s.h5'  %(fname, best_md5))
    pred_file = os.path.join(data_dir, bird, 'rnn', 'RNNLFPEncoderPred_%s_%s.h5'  %(fname, best_md5))
    linear_file = os.path.join(data_dir, bird, 'rnn', 'LFPEnvelope_%s.h5' % fname)

    if not os.path.exists(pred_file):
        rpt = RNNPreprocessTransform.load(preproc_file)
        rpt.write_pred_file(rnn_file, pred_file, lfp_enc_file=linear_file)
    visualize_pred_file(pred_file, 3, 0, 2.4, electrode_order=ROSTRAL_CAUDAL_ELECTRODES_LEFT[::-1], dbnoise=4.0)
    leg = custom_legend(['k', 'r', 'b'], ['Real', 'Linear', 'RNN'])
    plt.legend(handles=leg)

    fname = os.path.join(get_this_dir(), 'rnn_preds.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures():
    agg = RNNAggregator.load('/auto/tdrive/mschachter/data/aggregate/rnn.h5')

    # stats(agg)

    draw_preds(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()







