import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lasp.colormaps import plasma

from DecodingRhythms.utils import get_this_dir, clean_region, set_font
from zeebeez.aggregators.lfp_and_spike_psd_decoders import AggregateLFPAndSpikePSDDecoder
from zeebeez.utils import ROSTRAL_CAUDAL_ELECTRODES_LEFT, ROSTRAL_CAUDAL_ELECTRODES_RIGHT


def get_weights(agg, data_dir='/auto/tdrive/mschachter/data'):

    # construct a dataset for the weights
    weight_data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(),
                   'row':list(), 'col':list(), 'region':list(), 'weight':list(), 'r2':list(), 'aprop':list(),
                   'f':list()}

    # read electrode data
    edata = pd.read_csv(os.path.join(data_dir, 'aggregate', 'electrode_data.csv'))

    # read multiunit, single electrode, and single cell data
    df_me = pd.read_csv(os.path.join(data_dir, 'aggregate', 'multi_electrode_perfs.csv'))

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])
    print '# of groups: %d' % len(g)

    aprops = ['maxAmp', 'sal', 'meantime', 'entropytime', 'meanspect', 'q1', 'q2', 'q3', 'entropyspect']
    nbands = len(agg.freqs)

    all_weights = dict()

    decomps = ['locked', 'spike_psd']

    for (bird,block,segment,hemi),gdf in g:

        for decomp in decomps:

            index2electrode = agg.index2electrode[(bird,block,segment,hemi)]

            for aprop in aprops:

                # get blacklist of electrodes
                """
                i = (df_se.bird == bird) & (df_se.block == block) & (df_se.segment == segment) & (df_se.hemi == hemi)
                lkrat = df_se[i]['lkrat_%s' % aprop]
                gi = lkrat > 1
                good_electrodes = list(df_se[i][gi]['electrode'].values)
                """

                # get blacklist of bands
                i = (df_me.bird == bird) & (df_me.block == block) & (df_me.segment == segment) & (df_me.hemi == hemi) & (df_me.band > 0)
                lkrat = df_me[i]['lkrat_%s_lfp' % aprop]
                gi = lkrat > 1
                good_bands = list(df_me[i][gi]['band'].values)

                # grab the weights for the site
                i = (gdf.e1 == -1) & (gdf.e2 == -1) & (gdf.cell_index == -1) & (gdf.band == 0) & (gdf.exfreq == False) & \
                    (gdf.exel == False) & (gdf.aprop == aprop) & (gdf.decomp == decomp)

                assert i.sum() == 1

                # get the decoder performance
                r2 = gdf[i]['r2'].values[0]

                # get the weights
                windex = gdf[i]['weight_index'].values[0]
                wlen = gdf[i]['weight_length'].values[0]
                W = agg.acoustic_weights[windex:(windex+wlen), :]

                # get information about electrodes
                if hemi == 'L':
                    electrode_order = ROSTRAL_CAUDAL_ELECTRODES_LEFT
                else:
                    electrode_order = ROSTRAL_CAUDAL_ELECTRODES_RIGHT

                cell_i2e = np.array(agg.cell_index2electrode[(bird, block, segment, hemi)])

                # map indices to coordinates and regions
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

                if decomp == 'locked':
                    assert W.shape == (len(index2electrode), nbands)
                elif decomp == 'spike_psd':
                    assert W.shape == (len(cell_i2e), nbands)

                """
                # zero out bad frequency bands
                for b in range(1, nbands+1):
                    if b not in good_bands:
                        W[:, b-1] = 0.

                # zero out bad electrodes
                for k,e in enumerate(index2electrode):
                    if e not in good_electrodes:
                        W[k, :] = 0.
                """

                # zero out weights that are less than 1 SD
                wsd = np.abs(W).std(ddof=1)
                W[np.abs(W) < wsd] = 0.

                # reorder and/or reshape the weights
                Wrs = None
                Wro = None

                if decomp == 'spike_psd':
                    # reorder the cells by electrode
                    Wro = np.zeros_like(W)
                    the_index = 0
                    for e in electrode_order:
                        for k in np.where(cell_i2e == e)[0]:
                            Wro[the_index, :] = W[k, :]
                            the_index += 1

                elif decomp == 'locked':
                    # reshape matrix into (row,col,band)
                    Wrs = np.zeros([8, 2, nbands])
                    for k,(row,col) in enumerate(index2coord):
                        Wrs[row, col, :] = W[k, :]

                    # reorder the flat matrix
                    Wro = np.zeros_like(W)
                    for k,e in enumerate(electrode_order):
                        j = index2electrode.index(e)
                        Wro[k, :] = W[j, :]

                    # insert data into dataset
                    for k,e in enumerate(index2electrode):
                        for b,f in enumerate(agg.freqs):

                            # if e not in good_electrodes:
                            #     continue

                            # if b+1 not in good_bands:
                            #    continue

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

                all_weights[(bird, block, segment, hemi, aprop, decomp)] = (Wro, Wrs, electrode_order, index2electrode, cell_i2e, index2coord, index2label, r2)

    df_weight = pd.DataFrame(weight_data)
    df_weight.to_csv(os.path.join(data_dir, 'aggregate', 'weight_data.csv'), index=False)

    return all_weights, df_weight


def draw_some_weights(agg, data_dir='/auto/tdrive/mschachter/data'):

    birds = ['GreBlu9508M', 'YelBlu6903F', 'WhiWhi4522M']
    aprops = ['maxAmp', 'sal', 'meanspect', 'meantime', 'entropytime', 'q1', 'q2', 'q3', 'entropyspect']

    all_weights,df_weights = get_weights(agg, data_dir)

    weights_by_bird_and_hemi = dict()

    for (bird,block,segment,hemi,aprop,decomp),(Wro, Wrs, electrode_order, index2electrode, cell_i2e, index2coord, index2label, r2) in all_weights.items():
        if (bird, hemi, aprop,decomp) not in weights_by_bird_and_hemi:
            weights_by_bird_and_hemi[(bird, hemi, aprop,decomp)] = list()
        weights_by_bird_and_hemi[(bird, hemi, aprop, decomp)].append(Wro)

    """
    decomp = 'locked'
    for aprop in aprops:
        fig = plt.figure(figsize=(24, 12))
        nrows = len(birds)
        ncols = 2
        gs = plt.GridSpec(nrows, ncols)
        fig.subplots_adjust(top=0.95, bottom=0.01, hspace=0.25, wspace=0.25)

        for k,bird in enumerate(birds):

            Wlist_L = np.array(weights_by_bird_and_hemi[(bird, 'L', aprop, decomp)])
            Wlist_R = np.array(weights_by_bird_and_hemi[(bird, 'R', aprop, decomp)])

            Wleft = Wlist_L.mean(axis=0)
            Wright = Wlist_R.mean(axis=0)

            absmax = max(np.abs(Wleft).max(), np.abs(Wright).max())

            ax = plt.subplot(gs[k, 0])
            plt.imshow(Wleft, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='upper',
                       vmin=-absmax, vmax=absmax, extent=[agg.freqs.min(), agg.freqs.max(), 0, 16])
            plt.ylabel(bird)
            plt.xlabel('Frequency (Hz)')
            plt.title('Left Hemisphere')

            ax = plt.subplot(gs[k, 1])
            plt.imshow(Wright, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='upper',
                       vmin=-absmax, vmax=absmax, extent=[agg.freqs.min(), agg.freqs.max(), 0, 16])
            plt.ylabel(bird)
            plt.xlabel('Frequency (Hz)')
            plt.title('Right Hemisphere')

        plt.suptitle(aprop)

        fname = os.path.join(get_this_dir(), '%s_all_weights_%s.svg' % (decomp, aprop))
        print 'Saving %s' % fname
        plt.savefig(fname)
        plt.close('all')
    """

    decomp = 'spike_psd'
    for k,bird in enumerate(birds):

        for aprop in aprops:

            Wlist_L = np.array(weights_by_bird_and_hemi[(bird, 'L', aprop, decomp)])
            Wlist_R = np.array(weights_by_bird_and_hemi[(bird, 'R', aprop, decomp)])

            assert len(Wlist_L) == len(Wlist_R)
            nsites = len(Wlist_L)

            # create a figure
            fig = plt.figure(figsize=(24, 12))
            nrows = nsites
            ncols = 2
            gs = plt.GridSpec(nrows, ncols)
            fig.subplots_adjust(top=0.95, bottom=0.01, hspace=0.25, wspace=0.25)

            for n in range(nsites):
                Wleft = Wlist_L[n]
                Wright = Wlist_R[n]

                ncells_left = Wleft.shape[0]
                ncells_right = Wright.shape[0]

                absmax = max(np.abs(Wleft).max(), np.abs(Wright).max())

                ax = plt.subplot(gs[n, 0])
                plt.imshow(Wleft, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='upper',
                       vmin=-absmax, vmax=absmax, extent=[agg.freqs.min(), agg.freqs.max(), 0, ncells_left])
                plt.ylabel('Site %d' % (n+1))
                plt.xlabel('Frequency (Hz)')
                plt.title('Left Hemisphere')

                ax = plt.subplot(gs[n, 1])
                plt.imshow(Wright, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, origin='upper',
                           vmin=-absmax, vmax=absmax, extent=[agg.freqs.min(), agg.freqs.max(), 0, ncells_right])
                plt.ylabel('Site %d' % (n+1))
                plt.xlabel('Frequency (Hz)')
                plt.title('Right Hemisphere')

            plt.suptitle('%s: %s' % (bird, aprop))

            fname = os.path.join(get_this_dir(), '%s_%s_all_weights_%s.svg' % (decomp, bird, aprop))
            print 'Saving %s' % fname
            plt.savefig(fname)
            plt.close('all')


def draw_all_weights(agg, data_dir='/auto/tdrive/mschachter/data'):

    all_weights,df_weights = get_weights(agg, data_dir)

    for (bird,block,segment,hemi,aprop),(Wro, Wrs, electrode_order, index2electrode, index2coord, index2label, r2) in all_weights.items():

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

def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    agg_file = os.path.join(data_dir, 'aggregate', 'lfp_and_spike_psd_decoders.h5')
    agg = AggregateLFPAndSpikePSDDecoder.load(agg_file)

    # draw_all_weights(agg)
    draw_some_weights(agg)


if __name__ == '__main__':
    set_font()
    draw_figures()

