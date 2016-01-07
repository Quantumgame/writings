import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from zeebeez.transforms.biosound import BiosoundTransform
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform


def aggregate_gamma_data(bird='GreBlu9508M', block='Site4', segment='Call1', hemi='L', exclude_types=('wnoise', 'mlnoise'),
                         data_dir='/auto/tdrive/mschachter/data'):

    freq_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(),
                 'stim_id':list(), 'syllable_index':list(), 'electrode':list(), 'region':list(), 'cell_index':list(),
                 'lfp_freq':list(), 'lfp_freq_std':list(), 'spike_freq':list(), 'spike_freq_std':list(),
                 'spike_rate':list(), 'spike_rate_std':list()}

    acoustic_props = ['maxAmp', 'sal', 'meanspect', 'q1', 'q2', 'q3', 'entropyspect', 'meantime', 'entropytime']

    for aprop in acoustic_props:
        freq_data[aprop] = list()

    pdata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'protocol_data.csv'))

    gregs = pdata.groupby(['bird', 'block', 'segment'])

    for (bird,block,segment),greg_df in gregs:
        for hemi in ['L', 'R']:

            # load PairwiseCF transform
            tdir = os.path.join(data_dir, bird, 'transforms')
            pfile = os.path.join(tdir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird, block, segment, hemi))
            pt = PairwiseCFTransform.load(pfile)

            freqs = pt.freqs

            # load biosound
            bs_file = os.path.join(tdir, 'BiosoundTransform_%s.h5' % bird)
            bst = BiosoundTransform.load(bs_file)

            # map the biosound data to a dictionary for fast lookup
            bs_data = dict()
            i = np.ones(len(bst.stim_df), dtype='bool')
            for etype in exclude_types:
                i &= bst.stim_df['stim_type'] != etype
            df = bst.stim_df[i]
            g = df.groupby(['stim_id', 'order'])
            for (stim_id,order),gdf in g:
                assert len(gdf) == 1, "More than one entry in biosound df for stim id %d, order %d" % (stim_id, order)
                v = list()
                for aprop in acoustic_props:
                    v.append(gdf[aprop].values[0])
                bs_data[(stim_id, order)] = np.array(v)

            g = pt.df.groupby(['stim_id', 'order'])

            for (stim_id,order),gdf in g:
                # compute LFP center frequencies for each stim
                i = (gdf.decomp == 'locked') & (gdf.electrode1 == gdf.electrode2)
                assert i.sum() == 16
                electrodes = gdf.electrode1.unique()

                for e in electrodes:
                    # read the electrode PSD
                    i = (gdf.decomp == 'locked') & (gdf.electrode1 == e) & (gdf.electrode2 == e)
                    assert i.sum() == 1, "stim_id=%d, order=%d, e=%d, i.sum()=%d" % (stim_id, order, e, i.sum())

                    reg = gdf.region[i].values[0]
                    index = gdf['index'][i].values[0]

                    # get the LFP psd for this stim presentation
                    lfp_psd = pt.psds[index, :]
                    lfp_cfreq,lfp_cfreq_std = quantify_lfp(lfp_psd, freqs)

                    # read the single cell data
                    i = (gdf.decomp == 'spike_psd') & (gdf.electrode1 == e) & (gdf.electrode2 == e)
                    ncells = i.sum()
                    if ncells == 0:
                        print 'No cells for %s,%s,%s,%s, electrode=%d, stim_id=%d, order=%d' % \
                              (bird, block, segment, hemi, e, stim_id, order)
                        continue

                    cell_indices = gdf.cell_index[i].values
                    indices = gdf['index'].values
                    for ci,index in zip(cell_indices, indices):
                        # quantify the spike psd
                        spike_psd = pt.spike_psd[index, :]
                        spike_cfreq,spike_cfreq_std = quantify_lfp(spike_psd, freqs)

                        # get the mean and std spike rate
                        spike_rate,spike_rate_std = pt.spike_rate[index, :]

                        # insert the data into the dataset
                        freq_data['bird'].append(bird)
                        freq_data['block'].append(block)
                        freq_data['segment'].append(segment)
                        freq_data['hemi'].append(hemi)
                        freq_data['stim_id'].append(stim_id)
                        freq_data['syllable_index'].append(order)
                        freq_data['electrode'].append(e)
                        freq_data['region'].append(reg)
                        freq_data['cell_index'].append(ci)
                        freq_data['lfp_freq'].append(lfp_cfreq)
                        freq_data['lfp_freq_std'].append(lfp_cfreq_std)
                        freq_data['spike_freq'].append(spike_cfreq)
                        freq_data['spike_freq_std'].append(spike_cfreq_std)
                        freq_data['spike_rate'].append(spike_rate)
                        freq_data['spike_rate_std'].append(spike_rate_std)

                        for aprop,aval in zip(acoustic_props, bs_data[(stim_id,order)]):
                            freq_data[aprop].append(aval)

    # write to a file
    df = pd.DataFrame(freq_data)
    df.write_csv(os.path.join(data_dir, 'aggregate', 'gamma.csv'), index=False)


def quantify_lfp(lfp_psd, freqs):

    fi_gamma = (freqs > 0) & (freqs < 70) # 16-66Hz

    # quantify low gamma
    gfreq = lfp_psd[fi_gamma]
    gfreq /= gfreq.sum()

    # compute center frequency
    cfreq = np.sum(gfreq*freqs[fi_gamma])

    # compute std around center frequency
    cstd = np.sqrt(np.sum(gfreq * (freqs[fi_gamma] - cfreq)**2))

    return cfreq,cstd


if __name__ == '__main__':
    aggregate_gamma_data()



