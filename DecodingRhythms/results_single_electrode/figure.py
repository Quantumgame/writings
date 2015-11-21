import os
import h5py
import numpy as np
import operator
from numpy.fft import fftfreq
import pandas as pd

import matplotlib.pyplot as plt
from scipy.fftpack import fft
from lasp.plots import custom_legend
from utils import get_this_dir, set_font, add_region_info

from zeebeez.aggregators.pairwise_decoders_single import AggregatePairwiseDecoder
from zeebeez.utils import CALL_TYPES, REGION_NAMES_LONG, REGION_COLORS, DECODER_CALL_TYPES, REGION_NAMES, \
    CALL_TYPE_COLORS


def filter_bad_points(df, exclude_regions=True):

    exclude_i = np.zeros(len(df), dtype='bool')

    exclude_i |= np.isnan(df.pcc)

    exclude_i |= (df.bird == 'BlaBro09xxF') & (df.hemi == 'L')

    exclude_i |= (df.order == 'cross')
    exclude_i |= (df.order == 'self+cross')

    exclude_i |= (df.ptype == 'pair')

    if exclude_regions:
        exclude_i |= df.reg1 == 'HP'
        exclude_i |= df.reg1 == '?'
        exclude_i |= df.reg1 == 'L'
        exclude_i |= df.reg1.str.contains('-')

        exclude_i |= df.reg2 == 'HP'
        exclude_i |= df.reg2 == '?'
        exclude_i |= df.reg2 == 'L'
        exclude_i |= df.reg2.str.contains('-')

    print '%d of %d points excluded, %d points left' % (exclude_i.sum(), len(df), len(df) - exclude_i.sum())

    return df[~exclude_i]


def draw_good_weights(agg, df, thresh=0.15):

    # read frequencies from an arbitrary PairwiseCF file
    pcf_file = '/auto/tdrive/mschachter/data/GreBlu9508M/transforms/PairwiseCF_GreBlu9508M_Site4_Call1_L_raw.h5'
    hf = h5py.File(pcf_file, 'r')
    freqs = hf.attrs['freqs']
    hf.close()

    regs = ['L3', 'L1', 'L2', 'CMM', 'NCM', 'CML']

    weights_by_reg_and_decomp = dict()

    for reg in regs:
        for decomp in ['locked']:
            key = (reg, decomp)
            weights_by_reg_and_decomp[key] = list()

    nfreqs = len(freqs)

    print 'df='
    print df.head()

    print 'len(df)=%d' % len(df)

    # get all the weights
    weights = list()
    wkey = ('locked', 'self', 'single')
    i = (df.pcc > thresh) & (df.decomp == 'locked')
    print 'i.sum()=%d' % i.sum()
    indices = df['index'][i].values

    Wall = agg.weights[wkey][indices]
    print 'Wall.shape=',Wall.shape

    for W in Wall:
        assert np.sum(np.abs(W[0, :-1])) == 0, "Nonzero model weights for fixed term!"
        for w in W[1:, :-1]:
            weights.append(np.abs(w))

    weights = np.array(weights)
    print 'weights.shape=',weights.shape

    # flip weights so they all have positive cc
    w0 = weights[0]
    for k,w in enumerate(weights):
        cc = np.corrcoef(w0, w)[0, 1]
        if cc < 0:
            weights[k, :] = -w

    # compute the coefficient of variation for the weights
    wmean = weights.mean(axis=0)
    wstd = weights.std(axis=0, ddof=1)
    wcv = wstd / np.abs(wmean)

    figsize = (24, 16)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.40, wspace=0.20)

    ax = plt.subplot(2, 3, 1)
    for w in weights:
        plt.plot(freqs, np.abs(w), '-', linewidth=2.0, alpha=0.7)
    plt.axis('tight')

    ax = plt.subplot(2, 3, 4)
    # plot the weight CV
    print 'freqs.shape=',freqs.shape
    print 'wcv.shape=',wcv.shape
    plt.plot(freqs, wstd, 'k-', linewidth=4.0)
    # plt.plot(freqs, wcv, 'k-', linewidth=2.0, alpha=0.7)
    # plt.errorbar(freqs, wmean, yerr=wstd/np.sqrt(len(weights)), ecolor='r', color='k', elinewidth=3.0)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Standard Deviation')
    plt.title('Absolute Decoder Weight SD')
    plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'weights.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_perf_improvements(df):

    g = df.groupby(['bird', 'block', 'segment', 'hemi', 'e1'])

    total_diff = list()
    locked_diff = list()

    missing = list()

    for (bird,block,segment,hemi,e1),gdf in g:

        vals = dict()
        has_missing = False
        for d in ['total', 'locked', 'nonlocked']:
            i = gdf['decomp'] == d
            if i.sum() > 1:
                print gdf
            if i.sum() == 0:
                has_missing = True
                missing.append( (bird, block, segment, hemi, e1, d))
                continue
            # assert i.sum() == 1, "i.sum()=%d for %s,%s,%s,%s,%d, decomp=%s" % (i.sum(), bird, block, segment, hemi, e1, d)
            vals[d] = gdf['pcc'][i].values[0]

        if not has_missing:
            total_diff.append(vals['total'] - vals['locked'])
            locked_diff.append(vals['locked'] - vals['nonlocked'])

    print '# of missing files = %d:' % len(missing)
    print missing

    total_diff = np.array(total_diff)
    locked_diff = np.array(locked_diff)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(total_diff, bins=20)
    plt.title('total pcc - locked pcc')

    plt.subplot(2, 1, 2)
    plt.hist(locked_diff, bins=20)
    plt.title('locked pcc - nonlocked pcc')

    plt.axis('tight')


def draw_per_call_pcc_and_selectivity(agg, df, pcc_thresh=0.20):

    i = df.pcc > pcc_thresh
    gdf = df[i]
    
    # sort the call types by mean PCC
    ct_list = list()
    for k,ct in enumerate(DECODER_CALL_TYPES):
        key = 'pcc_%s' % ct
        ct_list.append( (ct, gdf[key].mean()))
    ct_list.sort(key=operator.itemgetter(-1))
    sorted_call_types = [x[0] for x in ct_list]

    region_order = ['L3', 'L1', 'L2', 'CMM', 'NCM', 'CML']

    decomps = ['locked']

    # get average confusion matrix
    ii = gdf['index'].values
    key = ('locked', 'self', 'single')
    C = agg.confidence_matrices[key][ii]
    nsamps,nc,nc = C.shape
    Cmean = np.zeros([nc, nc])
    for j in range(nsamps):
        cnames = list(agg.class_names[key][ii[j]])
        for m,ct1 in enumerate(DECODER_CALL_TYPES):
            for n,ct2 in enumerate(DECODER_CALL_TYPES):
                mm = cnames.index(ct1)
                nn = cnames.index(ct2)
                Cmean[m, n] += C[j, mm, nn]
    Cmean /= float(nsamps)

    # build a matrix of pcc and selectivity per call decomposition/type/region
    mats = dict()
    for decomp in decomps:
        mats[decomp] = dict()
        for val in ['pcc', 'sel']:
            mats[decomp][val] = np.zeros([len(DECODER_CALL_TYPES), len(REGION_NAMES)])

    for decomp in decomps:
        for k,ct in enumerate(sorted_call_types):
            for j,reg in enumerate(region_order):
                i = (gdf.reg1 == reg) & (gdf.decomp == decomp)
                for val in ['pcc', 'sel']:
                    key = '%s_%s' % (val, ct)
                    vals = gdf[i][key].values
                    mats[decomp][val][k, j] = vals.mean()

    # make plots
    figsize = (24, 8.3)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.97, left=0.03, hspace=0.20, wspace=0.45)

    nrows = 1
    ncols = 4

    gs = plt.GridSpec(nrows, ncols)
    # draw average confusion matrix
    ax = plt.subplot(gs[0, :2])
    plt.imshow(Cmean, cmap=plt.cm.afmhot, vmin=0, vmax=1, interpolation='nearest', aspect='auto', origin='lower')
    plt.xticks(range(len(DECODER_CALL_TYPES)), DECODER_CALL_TYPES)
    plt.yticks(range(len(DECODER_CALL_TYPES)), DECODER_CALL_TYPES)
    plt.colorbar()

    # draw pcc per region and selectivity
    lbls = {'pcc':'PCC', 'sel':'Selectivity'}

    for k,val in enumerate(['pcc', 'sel']):
        for j,decomp in enumerate(decomps):

            ax = plt.subplot(gs[0, k+2])

            plt.imshow(mats[decomp][val], interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot_r, origin='lower')
            plt.yticks(range(len(sorted_call_types)), sorted_call_types)
            plt.xticks(range(len(region_order)), region_order, rotation=45)
            plt.colorbar()
            plt.axis('tight')
            plt.title(lbls[val])
            plt.ylabel('Call Type')
            plt.xlabel('Region')

    fname = os.path.join(get_this_dir(), 'pcc_selectivity_by_call_type_and_region.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_region_boxplot(df):
    # make a boxplot of pcc per region
    perf_list = list()
    regs = df.reg1.unique()
    for reg in regs:
        i = df.reg1 == reg
        vals = df.pcc[i].values
        median_pcc = np.median(vals)
        perf_list.append((reg, vals, median_pcc))
    perf_list.sort(key=operator.itemgetter(-1))

    key_order = [x[0] for x in perf_list]
    vals = [x[1] for x in perf_list]
    figsize = (16, 8)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    bp = plt.boxplot(vals)
    plt.setp(bp['boxes'], lw=0, color='k')
    plt.setp(bp['whiskers'], lw=3.0, color='k')
    # plt.xticks(range(1, len(key_order)+1), key_order, rotation=60)
    plt.xticks(range(1, len(key_order) + 1), key_order)
    for k, stim_class in enumerate(key_order):
        box = bp['boxes'][k]
        boxX = list()
        boxY = list()
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = zip(boxX, boxY)
        boxPolygon = plt.Polygon(boxCoords, facecolor='#663333')
        ax.add_patch(boxPolygon)

        med = bp['medians'][k]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k', linewidth=3.0)

        plt.plot([np.average(med.get_xdata())], [np.average(perf_list[k][-1])], color='w', marker='*',
            markeredgecolor='k', markersize=15)

    plt.ylim(0.0, 0.5)
    plt.ylabel('PCC')


def draw_perf_bars(df):

    # rank regions by mean PCC for self+cross models
    pcc_by_region = dict()
    regs = df.reg1.unique()
    pcc_region_order = list()
    for reg in regs:
        pcc_by_order = dict()
        for d in ['self+cross', 'self', 'cross']:
            i = (df.reg1 == reg) & (df.order == d)
            pcc_by_order[d] = df[i].pcc.values
        pcc_by_region[reg] = pcc_by_order

        pcc_region_order.append( (reg, pcc_by_order['self+cross'].mean()) )

    pcc_region_order.sort(key=operator.itemgetter(1))

    figsize = (24, 13)
    fig = plt.figure(figsize=figsize)

    space_per_reg = 5

    # make a list of data to be plotted
    x = list()
    y = list()
    err = list()
    xticks = list()
    colors = list()

    c_by_decomp = {'self+cross':'#333333', 'self':'#CC3333', 'cross':'#0033CC'}

    for k,(reg,pcc_mean) in enumerate(pcc_region_order):
        pcc_by_order = pcc_by_region[reg]

        for j,decomp in enumerate(['self+cross', 'self', 'cross']):
            xloc = k*space_per_reg + j + 1
            x.append(xloc)
            y.append(pcc_by_order[decomp].mean())
            err.append(pcc_by_order[decomp].std(ddof=1))
            colors.append(c_by_decomp[decomp])

            if j == 1:
                xticks.append( (xloc, reg) )

    ax = plt.subplot(111)
    ax.bar(x, y, color=colors, yerr=err, ecolor='k')
    plt.xticks([xx[0] for xx in xticks], [xx[1] for xx in xticks])
    plt.ylim(0, 0.75)
    leg = custom_legend([c_by_decomp['self+cross'], c_by_decomp['self'], c_by_decomp['cross']], ['self+cross', 'self', 'cross'])
    plt.legend(handles=leg, fontsize='small')
    plt.xlabel('Region')
    plt.ylabel('PCC')

    fname = os.path.join(get_this_dir(), 'perf_bar_by_region.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def export_csv(df):
    ofile = '/auto/tdrive/mschachter/data/aggregate/glmm_correlation_single.csv'

    export_data = dict()
    cols = ['bird', 'hemi', 'order', 'decomp', 'pcc', 'gs']
    for c in cols:
        export_data[c] = list()

    for c in df.keys():
        if c.startswith('pcc_') or c.startswith('sel_'):
            export_data[c] = list()

    export_data['site'] = list()
    export_data['protocol'] = list()
    export_data['region'] = list()
    export_data['electrode'] = list()

    i = df.ptype == 'single'
    for k,row in df[i].iterrows():

        site = '%s_%s_%s' % (row['bird'], row['block'], row['hemi'])
        protocol = row['segment']

        export_data['site'].append(site)
        export_data['protocol'].append(protocol)
        export_data['region'].append(row['reg1'])
        export_data['electrode'].append(row['e1'])

        for c in df.keys():
            if c.startswith('pcc_') or c.startswith('sel_'):
                export_data[c].append(row[c])

        for c in cols:
            export_data[c].append(row[c])

    export_df = pd.DataFrame(export_data)
    export_df.to_csv(ofile, index=False)


def draw_figures():

    agg_file = '/auto/tdrive/mschachter/data/aggregate/decoders_pairwise_coherence_single.h5'
    agg = AggregatePairwiseDecoder.load(agg_file)

    print 'len(df)(1)=%d' % len(agg.df)
    df,class_names = add_region_info(agg, agg.df)
    print 'len(df)(2)=%d' % len(df)
    df = filter_bad_points(df)
    print 'len(df)(3)=%d' % len(df)

    export_csv(df)

    uregs = df.reg1.unique()
    for reg in uregs:
        i = (df.reg1 == reg) & (df.decomp == 'locked')
        print '# of samples in %s: %d' % (reg, i.sum())

    print 'types=',df.order.unique()

    # draw_perf_bars(df)

    ix = (df.decomp == 'locked') & (df.order == 'self')
    draw_region_boxplot(df[ix])
    fname = os.path.join(get_this_dir(), 'self_locked_pcc_by_region.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')
    # plt.show()

    draw_good_weights(agg, df[ix])

    # plt.show()

    draw_per_call_pcc_and_selectivity(agg, df[ix])
    plt.show()

    return

    draw_perf_improvements(df)

    for decomp in ['total', 'locked', 'nonlocked']:

        i = (df.ptype == 'single') & (df.order == 'self') & (df.decomp == decomp)
        print '# of single data points (%s): %d' % (decomp, i.sum())

        draw_region_boxplot(df[i])
        plt.title('Single Electrode Decoder Performance by Region (%s)' % decomp)
        fname = os.path.join(get_this_dir(), 'perf_boxplot_by_region_%s.svg' % decomp)
        plt.savefig(fname, facecolor='w', edgecolor='none')


if __name__ == '__main__':

    set_font()

    draw_figures()
    plt.show()
