import os
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import set_font, get_this_dir
from zeebeez.aggregators.rnn import RNNAggregator


def stats(agg):
    data = {'bird': list(), 'block': list(), 'segment': list(), 'hemi': list(), 'electrode': list(),
            'linear_cc': list(), 'cc': list(), 'err': list(),
            'lambda1': list(), 'lambda2': list(), 'n_unit': list()}

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

    df = pd.DataFrame(data)
    df.to_csv('/auto/tdrive/mschachter/data/aggregate/rnn_best.csv', header=True, index=False)

    fig = plt.figure(figsize=(12, 10))
    x = np.linspace(0, 1, 20)
    plt.plot(x, x, 'k-')
    plt.plot(df.linear_cc, df.cc, 'go', alpha=0.7, markersize=12)
    plt.xlabel('Linear CC')
    plt.ylabel('RNN CC')
    plt.xlim(0, 0.8)
    plt.ylim(0, 0.8)

    fname = os.path.join(get_this_dir(), 'linear_vs_rnn_cc.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()


def draw_figures():
    agg = RNNAggregator.load('/auto/tdrive/mschachter/data/aggregate/rnn.h5')

    stats(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()







