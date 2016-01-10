import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import COLOR_BLUE_LFP, COLOR_YELLOW_SPIKE, set_font


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    df = pd.read_csv(os.path.join(data_dir, 'aggregate', 'gamma.csv'))

    i = (df.bird != 'BlaBro09xxF') & (df.spike_rate > 0) & (df.spike_freq > 0) & (df.spike_freq_std > 0) & (df.region != '?') & ~df.region.str.contains('-')
    print '# of data points: %d' % i.sum()

    # plot histograms of LFP frequency and spike frequency
    plt.figure()

    plt.hist(df[i].lfp_freq.values, bins=30, color=COLOR_BLUE_LFP, alpha=0.75, normed=True)
    plt.hist(df[i].spike_freq.values, bins=30, color=COLOR_YELLOW_SPIKE, alpha=0.75, normed=True)

    # ax = plt.gca()
    # ax.set_yscale('log', basey=2)

    plt.legend(['LFP', 'Spike PSD'])
    plt.xlabel('Low Gamma Center Frequency (Hz)')
    plt.ylabel('Proportion of Cells')
    plt.axis('tight')
    plt.xlim(10, 65)

    plt.show()


if __name__ == '__main__':
    set_font()
    draw_figures()



