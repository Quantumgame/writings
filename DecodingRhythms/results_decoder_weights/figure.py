import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from DecodingRhythms.utils import get_this_dir, clean_region
from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)

    # construct a dataset for the weights
    weight_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(),
                   'row':list(), 'col':list(), 'region':list(), 'weight':list(), 'r2':list(), 'aprop':list(),
                   'f':list()}

    # read electrode data
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    # read multiunit, single electrode, and single cell data
    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))
    df_se = pd.read_csv(os.path.join(data_dir, 'aggregate', 'single_electrode_perfs.csv'))
    df_cell = pd.read_csv(os.path.join(data_dir, 'aggregate', 'cell_perfs.csv'))

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])
    print '# of groups: %d' % len(g)

    aprops = ['maxAmp', 'sal', 'meantime', 'entropytime', 'meanspect', 'q1', 'q2', 'q3', 'entropyspect']
    nbands = len(agg.freqs)

    for (bird,block,segment,hemi),gdf in g:

        index2electrode = agg.index2electrode[(bird,block,segment,hemi)]

        for aprop in aprops:

            # get blacklist of electrodes
            i = (df_se.bird == bird) & (df_se.block == block) & (df_se.segment == segment) & (df_se.hemi == hemi)
            lkrat = df_se[i]['lkrat_%s' % aprop]
            gi = lkrat > 1
            good_electrodes = list(df_se[i][gi]['electrode'].values)

            # get blacklist of bands
            i = (df_me.bird == bird) & (df_me.block == block) & (df_me.segment == segment) & (df_me.hemi == hemi) & (df_me.band > 0)
            lkrat = df_me[i]['lkrat_%s_lfp' % aprop]
            gi = lkrat > 1
            good_bands = list(df_me[i][gi]['band'].values)

            # grab the weights for the site
            i = (gdf.e1 == -1) & (gdf.e2 == -1) & (gdf.cell_index == -1) & (gdf.band == 0) & (gdf.exfreq == False) & \
                (gdf.exel == False) & (gdf.aprop == aprop) & (gdf.decomp == 'locked')

            assert i.sum() == 1

            r2 = gdf[i]['r2'].values[0]

            windex = gdf[i]['weight_index'].values[0]
            wlen = gdf[i]['weight_length'].values[0]
            W = agg.acoustic_weights[windex:(windex+wlen), :]
            assert W.shape == (len(index2electrode), nbands)

            # zero out bad frequency bands
            for b in range(1, nbands+1):
                if b not in good_bands:
                    W[:, b-1] = 0.

            # zero out bad electrodes
            for k,e in enumerate(index2electrode):
                if e not in good_electrodes:
                    W[k, :] = 0.

            # reshape matrix into (row,col,band)
            Wrs = np.zeros([8, 2, nbands])
            index2coord = list()
            index2label = list()
            for k,e in enumerate(index2electrode):
                i = (edata.bird == bird) & (edata.block == block) & (edata.hemisphere == hemi) & (edata.electrode == e)
                assert i.sum() == 1
                row = edata[i]['row'].values[0]
                col = edata[i]['col'].values[0]
                reg = edata[i]['region'].values[0]
                index2coord.append((row, col))
                index2label.append('%d\n%s' % (e, reg))
                Wrs[row, col, :] = W[k, :]

                for b,f in enumerate(agg.freqs):

                    if e not in good_electrodes:
                        continue

                    if b+1 not in good_bands:
                        continue

                    weight_data['bird'].append(bird)
                    weight_data['block'].append(block)
                    weight_data['segment'].append(segment)
                    weight_data['hemi'].append(hemi)
                    weight_data['electrode'].append(e)
                    weight_data['row'].append(row + 1)

                    col_name = 'medial'
                    if hemi == 'L' and col == 0:
                        col_name = 'lateral'
                    if hemi == 'R' and col == 1:
                        col_name = 'lateral'

                    weight_data['col'].append(col_name)

                    weight_data['region'].append(clean_region(reg))
                    weight_data['r2'].append(r2)
                    weight_data['aprop'].append(aprop)
                    weight_data['f'].append(f)
                    weight_data['weight'].append(W[k, b])

            # reorder the flat matrix
            if hemi == 'L':
                electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
            else:
                electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

            Wro = np.zeros_like(W)
            for k,e in enumerate(electrode_order):
                j = index2electrode.index(e)
                Wro[k, :] = W[j, :]

            """
            # plot the decoder weights as a matrix
            fig = plt.figure(figsize=(12, 8))
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.30, wspace=0.20)
            absmax = np.abs(Wro).max()
            plt.imshow(Wro, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, origin='upper',
                       cmap=plt.cm.seismic)
            ylbls = ['%d' % e for e in electrode_order]
            plt.yticks(np.arange(len(electrode_order)), ylbls)
            plt.xticks(np.arange(len(agg.freqs)), ['%d' % f for f in agg.freqs])
            plt.ylabel('Electrode')
            plt.xlabel('Frequency (Hz)')
            fname = '%s_%s_%s_%s_%s' % (aprop, bird, hemi, block, segment)
            plt.suptitle('%s %0.2f' % (fname, r2))

            fname = os.path.join(get_this_dir(), 'weights', 'flat_%s.svg' % fname)
            print 'Saving %s' % fname
            plt.savefig(fname, facecolor='w', edgecolor='none')
            """

            """
            # plot the decoder weights spatially
            fig = plt.figure(figsize=(24, 13))
            plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.30, wspace=0.20)

            nrows = 2
            ncols = 6
            absmax = np.abs(Wrs).max()
            for k,f in enumerate(agg.freqs):
                W = Wrs[:, :, k]
                ax = plt.subplot(nrows, ncols, k+1)
                plt.imshow(W, origin='upper', interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax,
                           cmap=plt.cm.seismic, extent=[0, 2, 0, 8])

                for txt,(row,col) in zip(index2label, index2coord):
                    plt.text(col + 0.5, (7 - row) + 0.25, txt, horizontalalignment='center', color='k', fontsize=14)

                # plt.colorbar()
                plt.title('%d Hz' % f)
                plt.xticks([])
                plt.yticks([])

            fname = '%s_%s_%s_%s_%s' % (aprop, bird, hemi, block, segment)
            plt.suptitle('%s %0.2f' % (fname, r2))
            fname = os.path.join(get_this_dir(), 'weights', 'weights_%s.svg' % fname)

            print 'Saving %s' % fname
            plt.savefig(fname, facecolor='w', edgecolor='none')
            plt.close('all')
            """

    df_weight = pd.DataFrame(weight_data)
    df_weight.to_csv(os.path.join(data_dir, 'aggregate', 'weight_data.csv'), index=False)


if __name__ == '__main__':
    draw_figures()