import os
import operator

import numpy as np
import matplotlib.pyplot as plt

from zeebeez.aggregators.tuning_curve import TuningCurveAggregator


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'tuning_curve.h5')
    agg = TuningCurveAggregator.load(agg_file)

    i = (agg.df.decomp == 'full_psds') & (agg.df.freq == 99) & (agg.df.aprop == 'stdtime')
    assert i.sum() > 0

    perf = agg.df.mse_nonlin[i].values / agg.df.mse_mean[i].values
    perf[np.isnan(perf)] = 1.
    perf[np.isinf(perf)] = 1.

    perf_thresh = 0.95
    pi = perf < perf_thresh
    xindex = agg.df.xindex[i][pi]
    assert len(xindex) > 0

    lst = zip(xindex, perf[pi])
    lst.sort(key=operator.itemgetter(1))
    xindex = [x[0] for x in lst]

    cx = agg.curve_x[xindex, :]
    tc = agg.tuning_curves[xindex, :]

    plt.figure()
    for x,y in zip(cx,tc):
        plt.plot(x, y, 'k-', alpha=0.7)
    plt.xlabel('stdtime')
    plt.ylabel('Power (99Hz)')

    plt.show()


if __name__ == '__main__':
    draw_figures()
