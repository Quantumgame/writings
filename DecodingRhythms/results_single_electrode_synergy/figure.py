import os
import numpy as np

import matplotlib.pyplot as plt
import operator

from zeebeez.aggregators.pairwise_decoders_single import AggregatePairwiseDecoder

from utils import get_this_dir, set_font, add_region_info


def filter_bad_points(df, exclude_regions=True):

    exclude_i = np.zeros(len(df), dtype='bool')

    exclude_i |= np.isnan(df.pcc)

    exclude_i |= (df.bird == 'BlaBro09xxF') & (df.hemi == 'L')

    exclude_i |= (df.order == 'cross')
    exclude_i |= (df.order == 'self+cross')

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


def draw_synergy_hists(agg, df):

    # group by every pair within a site
    isum = df.ptype == 'single'
    print 'isum=%d' % isum.sum()
    i = df.ptype == 'pair'
    g = df[i].groupby(['bird', 'block', 'segment', 'hemi', 'e1', 'e2'])

    all_pccs = dict()
    for decomp in ['total', 'locked', 'nonlocked']:
        all_pccs[decomp] = list()

    pcc_change_by_regions = dict()

    for (bird,block,segment,hemi,e1,e2),gdf in g:

        for decomp in ['locked']:

            # get the single electrode performances
            i1 = (df.decomp == decomp) & (df.ptype == 'single') & (df.bird == bird) & \
                 (df.block == block) & (df.segment == segment) & (df.hemi == hemi) & (df.e1 == e1) & (df.e2 == e1)
            if i1.sum() == 0:
                print 'Missing: (%s, %s, %s, %s, %s, %d)' % (bird, block, segment, hemi, decomp, e1)
                continue
            assert i1.sum() == 1

            pcc1 = df.pcc[i1].values[0]

            i2 = (df.decomp == decomp) & (df.ptype == 'single') & (df.bird == bird) & \
                 (df.block == block) & (df.segment == segment) & (df.hemi == hemi) & (df.e2 == e2) & (df.e1 == e2)
            if i2.sum() == 0:
                print 'Missing: (%s, %s, %s, %s, %s, %d)' % (bird, block, segment, hemi, decomp, e2)
                continue
            assert i2.sum() == 1

            pcc2 = df.pcc[i2].values[0]

            # get the dual-electrode performance
            i12 = (gdf.decomp == decomp)
            assert i12.sum() == 1, "i12.sum()=%d" % i12.sum()

            pcc12 = gdf.pcc[i12].values[0]

            mpcc = max(pcc1, pcc2)

            all_pccs[decomp].append( (pcc1, pcc2, mpcc, pcc12) )

            reg1 = gdf[i12].reg1.values[0]
            reg2 = gdf[i12].reg2.values[0]
            rpair = tuple(sorted((reg1, reg2)))
            if rpair not in pcc_change_by_regions:
                pcc_change_by_regions[rpair] = list()
            pcc_change_by_regions[rpair].append(pcc12 - mpcc)

    figsize = (23, 10)
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.99, hspace=0.20, wspace=0.20)

    max_pcc = 0.7
    decomp = 'locked'

    pccs = np.array(all_pccs[decomp])
    max_pccs = pccs[:, 2]
    dual_pccs = pccs[:, 3]

    # organize the change in performance per region pair
    rlist = list()
    for rp,vals in pcc_change_by_regions.items():
        xpcc_mean = np.mean(vals)
        xpcc_std = np.std(vals, ddof=1)
        rlist.append((rp, xpcc_mean, xpcc_std))

    rlist.sort(key=operator.itemgetter(1))
    rpairs_ordered = [x[0] for x in rlist]
    nsamps = len(rlist)
    xpcc_means = np.array([x[1] for x in rlist])
    xpcc_stds = np.array([x[2] for x in rlist])

    ax = plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, max_pcc, 20), np.linspace(0, max_pcc, 20), 'k-', linewidth=2.0)
    plt.plot(max_pccs, dual_pccs, 'o', c='k', alpha=0.75)
    plt.title(decomp)
    plt.xlabel('Max PCC of Individual')
    plt.ylabel('Joint PCC')
    plt.xlim(0, max_pcc)
    plt.ylim(0, max_pcc)

    ind = np.arange(len(rpairs_ordered))
    ax = plt.subplot(1, 2, 2)
    plt.axvline(0, c='k')
    ax.bar(0, 0.8, width=xpcc_means, bottom=ind, xerr=xpcc_stds/np.sqrt(nsamps), facecolor='#6600FF', align='center',
           ecolor='k', orientation='horizontal', capsize=5., alpha=0.75, linewidth=3.0)
    plt.yticks(ind, ['%s-%s' % (r1,r2) for r1,r2 in rpairs_ordered], rotation=0)
    plt.xlabel('Joint PCC - Max PCC of Individual')
    plt.axis('tight')

    fname = os.path.join(get_this_dir(), 'synergy.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')


def draw_figures():

    agg_file = '/auto/tdrive/mschachter/data/aggregate/decoders_pairwise_coherence_single.h5'
    agg = AggregatePairwiseDecoder.load(agg_file)

    print 'len(df)(1)=%d' % len(agg.df)
    df,class_names = add_region_info(agg, agg.df)
    print 'len(df)(2)=%d' % len(df)
    df = filter_bad_points(df)
    print 'len(df)(3)=%d' % len(df)

    draw_synergy_hists(agg, df)


if __name__ == '__main__':

    set_font()
    draw_figures()
    plt.show()
