import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    pcf_file = os.path.join(data_dir, 'GreBlu9508M', 'transforms', 'PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5')
    pcf = PairwiseCFTransform.load(pcf_file)

    lags_index = np.abs(pcf.lags) == 0.

    i = pcf.df.stim_type != 'mlnoise'
    df = pcf.df[i]
    
    # transform psds
    psds = deepcopy(pcf.psds)
    pcf.log_transform(psds)
    # psds -= psds.mean(axis=0)
    # psds /= psds.std(axis=0, ddof=1)

    # transform pairwise cfs
    cross_cfs = pcf.cross_cfs[:, lags_index]
    
    # transform synchrony
    syncs = deepcopy(pcf.spike_synchrony)
    # sync -= sync.mean(axis=0)
    # sync /= sync.std(axis=0, ddof=1)

    X = list()

    electrodes = pcf.df.electrode1.unique()

    g = df.groupby(['stim_id', 'order'])
    for (stim_id,syllable_order),gdf in g:

        for k,e1 in enumerate(electrodes):
            for j in range(k):
                e2 = electrodes[j]

                i = (gdf.decomp == 'locked') & (gdf.electrode1 == e1) & (gdf.electrode2 == e2)
                if i.sum() == 0:
                    i = (gdf.decomp == 'locked') & (gdf.electrode1 == e2) & (gdf.electrode2 == e1)
                assert i.sum() == 1
                indices = gdf['index'][i]
                cf = cross_cfs[indices, :]
                cf_sum = np.abs(cf).sum()

                # compute average spike synchrony for this stim and electrode
                i = (gdf.decomp == 'spike_sync') & ((gdf.electrode1 == e1) | (gdf.electrode2 == e2))
                if i.sum() == 0:
                    i = (gdf.decomp == 'spike_sync') & ((gdf.electrode1 == e2) | (gdf.electrode2 == e1))
                assert i.sum() >= 1
                indices = gdf['index'][i]
                sync12 = syncs[indices].max()

                if cf_sum > 0 and sync12 > 0:
                    X.append((cf_sum, sync12))

    X = np.array(X)
    cc = np.corrcoef(X[:, 0], X[:, 1])[0, 1]

    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.plot(X[:, 1], X[:, 0], 'go')
    plt.xlabel('Spike Synchrony')
    plt.ylabel('LFP Synchrony (-20ms to 20ms)')
    plt.title('cc=%0.2f' % cc)
    plt.axis('tight')

    plt.show()

if __name__ == '__main__':

    draw_figures()

