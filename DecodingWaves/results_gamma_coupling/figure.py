import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import set_font

from zeebeez.aggregators.pop_spike_and_lfp import AggregatePopSpikeAndLFP


def is_bad_reg(reg):
    if '-' in reg:
        return True
    if reg in ['?', 'HP']:
        return True

    return False

def clean_reg(reg):
    if reg.startswith('L2'):
        return 'L2'
    return reg


def get_electrode_data():
    edata = pd.read_csv('/auto/tdrive/mschachter/data/aggregate/electrode_data.csv')

    electrode2region = dict()
    electrode2coord = dict()
    g = edata.groupby(['bird', 'block', 'hemisphere', 'electrode'])
    for (bird,block,hemi,e),gdf in g:
        assert len(gdf) == 1
        reg = gdf.region.values[0]
        row = gdf['row'].values[0]
        col = gdf['col'].values[0]
        electrode2region[(bird, block, hemi, e)] = reg
        electrode2coord[(bird, block, hemi, e)] = (row, col)

    return electrode2coord,electrode2region


def make_coupling_df(agg):

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode1':list(), 'electrode2':list(),
            'region1':list(), 'region2':list(), 'silent_weight':list(), 'evoked_weight':list(), 'distance':list()}

    electrode2coord,electrode2region = get_electrode_data()

    assert isinstance(agg, AggregatePopSpikeAndLFP)

    g = agg.df.groupby(['bird', 'block', 'segment', 'hemi'])
    for (bird,block,seg,hemi),gdf in g:

        assert len(gdf) == 1
        index = gdf['index'].values[0]
        index2electrode = agg.index2electrode[index]
        Wsilent = agg.W_silent_gamma[index]
        Wevoked = agg.W_evoked_gamma[index]

        for n1,e1 in enumerate(index2electrode):
            for n2 in range(n1):
                e2 = index2electrode[n2]

                reg1 = clean_reg(electrode2region[(bird,block,hemi,e1)])
                reg2 = clean_reg(electrode2region[(bird,block,hemi,e2)])

                row1,col1 = electrode2coord[(bird,block,hemi,e1)]
                row2,col2 = electrode2coord[(bird,block,hemi,e2)]

                medlat_dist = (col2-col1)*500.
                rostcaud_dist = (row2-row1)*250.
                dist = int(np.sqrt(medlat_dist**2 + rostcaud_dist**2))

                if is_bad_reg(reg1) or is_bad_reg(reg2):
                    continue

                ws = Wsilent[n1, n2]
                we = Wevoked[n1, n2]

                data['bird'].append(bird)
                data['block'].append(block)
                data['segment'].append(seg)
                data['hemi'].append(hemi)
                data['electrode1'].append(e1)
                data['electrode2'].append(e2)
                data['region1'].append(reg1)
                data['region2'].append(reg2)
                data['silent_weight'].append(ws)
                data['evoked_weight'].append(we)
                data['distance'].append(dist)

    return pd.DataFrame(data)


def compute_hist_stats(hist):

    p = hist / hist.sum()
    nbins = len(p)
    theta = np.linspace(-np.pi, np.pi, nbins)
    deg = ((180. * theta) / np.pi) + 180.

    stats = dict()

    avgsin = np.sum(p*np.sin(theta))
    avgcos = np.sum(p*np.cos(theta))
    stats['mean'] = ((180. * np.arctan2(avgsin, avgcos)) / np.pi) + 180.

    stats['std'] = np.sqrt(np.sum(p*(deg - stats['mean'])**2))
    stats['kurtosis'] = np.sum(p*(deg - stats['mean'])**4) / stats['std']**4

    maxent = np.log2(nbins)
    nzna = ~np.isnan(p) & (p > 0)
    stats['entropy'] = -np.sum(p[nzna]*np.log2(p[nzna])) / maxent

    return stats


def make_phase_df(agg):

    assert isinstance(agg, AggregatePopSpikeAndLFP)

    data = {'bird':list(), 'block':list(), 'segment':list(), 'hemi':list(), 'electrode':list(), 'region':list(),
            'evoked_kurtosis':list(), 'evoked_entropy':list(), 'evoked_mean':list(), 'evoked_std':list(),
            'silent_kurtosis':list(), 'silent_entropy':list(), 'silent_mean':list(), 'silent_std':list(),
            'hist_index':list()}

    electrode2coord,electrode2region = get_electrode_data()

    for ri,row in agg.df.iterrows():
        index = row['index']
        hist_index = row['hist_index']
        ncells = row['ncells']

        ei = hist_index + ncells

        evoked_hists = agg.phase_hist_evoked_gamma[hist_index:ei]
        silent_hists = agg.phase_hist_silent_gamma[hist_index:ei]
        cell_index2electrode = agg.cell_electrode[hist_index:ei]

        # one row for each cell
        for n,e in enumerate(cell_index2electrode):

            reg = clean_reg(electrode2region[(row['bird'], row['block'], row['hemi'], e)])
            if is_bad_reg(reg):
                continue

            for k in ['bird', 'block', 'segment', 'hemi']:
                data[k].append(row[k])

            # compute stats on the evoked and silent phase histograms
            ehist = evoked_hists[n]
            shist = silent_hists[n]
            for c,hist in zip(['evoked', 'silent'], (ehist, shist)):
                stats = compute_hist_stats(hist)
                for sname,sval in stats.items():
                    dname = '%s_%s' % (c, sname)
                    data[dname].append(sval)

            data['electrode'].append(e)
            data['region'].append(reg)
            data['hist_index'].append(hist_index + n)

    return pd.DataFrame(data)


def plot_coupling_stats(cdf):

    q99 = np.percentile(cdf.evoked_weight, 99)

    unique_dists = sorted(np.unique(cdf.distance))

    evoked_weight_by_dist = list()
    silent_weight_by_dist = list()

    for ud in unique_dists:
        i = cdf.distance == ud

        ew_mean = cdf.evoked_weight[i].mean()
        ew_std = cdf.evoked_weight[i].std(ddof=1)

        sw_mean = cdf.silent_weight[i].mean()
        sw_std = cdf.silent_weight[i].std(ddof=1)

        evoked_weight_by_dist.append( (ew_mean, ew_std))
        silent_weight_by_dist.append( (sw_mean, sw_std))

    evoked_weight_by_dist = np.array(evoked_weight_by_dist)
    silent_weight_by_dist = np.array(silent_weight_by_dist)

    nsamps = len(evoked_weight_by_dist)

    fig = plt.figure()

    ax = plt.subplot(1, 2, 1)
    plt.hist(cdf.evoked_weight, bins=70, color='r', alpha=0.7)
    plt.hist(cdf.silent_weight, bins=70, color='#303030', alpha=0.7)
    plt.legend(['Evoked', 'Spontaneous'])
    plt.xlabel('Coupling Weight')
    plt.axis('tight')
    plt.xlim(0, q99)

    ax = plt.subplot(1, 2, 2)
    plt.errorbar(unique_dists, evoked_weight_by_dist[:, 0], yerr=evoked_weight_by_dist[:, 1]/np.sqrt(nsamps), c='r', linewidth=7.0, alpha=0.9)
    plt.errorbar(unique_dists, silent_weight_by_dist[:, 0], yerr=silent_weight_by_dist[:, 1]/np.sqrt(nsamps), c='#303030', linewidth=7.0, alpha=0.7)
    plt.xlabel('Distance ($\mu$m)')
    plt.ylabel('Weight')
    plt.axis('tight')
    plt.legend(['Evoked', 'Spontaneous'])


def plot_hists(agg, pdf, sort_term='silent_entropy'):

    pdf = pdf.sort([sort_term], ascending=True)
    lst = zip(pdf[sort_term].values, pdf.hist_index)

    fig = plt.figure()
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05, hspace=0.20, wspace=0.20)
    nrows = 1
    ncols = 3

    # plot mean angle of top 100
    mean_angle_top100 = np.mean(pdf.silent_mean.values[:100])
    print 'mean angle: %f' % mean_angle_top100
    print '# of hists: %d' % len(lst)

    list_indices = [25, 200, 400]

    for k,li in enumerate(list_indices):

        kval,hi = lst[li]

        ehist = agg.phase_hist_evoked_gamma[hi]
        shist = agg.phase_hist_silent_gamma[hi]
        nbins = len(ehist)
        bwidth = 360. / nbins

        ax = plt.subplot(nrows, ncols, k+1)
        center = np.linspace(0, 360, nbins)
        plt.bar(center, shist, align='center', width=bwidth, color='#303030', alpha=0.7)
        plt.bar(center, ehist, align='center', width=bwidth, color='r', alpha=0.45)
        plt.xlabel('Phase Angle (deg)')
        plt.ylabel('Normalized Count')
        plt.axis('tight')
        plt.title('%0.3f' % kval)

    plt.show()


def draw_figures():

    agg = AggregatePopSpikeAndLFP.load('/auto/tdrive/mschachter/data/aggregate/pop_spike_and_lfp.h5')

    cdf = make_coupling_df(agg)
    cdf.to_csv('/auto/tdrive/mschachter/data/aggregate/coupling.csv', index=False)

    pdf = make_phase_df(agg)
    pdf.to_csv('/auto/tdrive/mschachter/data/aggregate/spike_phase.csv', index=False)

    plot_hists(agg, pdf)

    # plot_coupling_stats(cdf)
    # plot_phase_stats(pdf)

    plt.show()


if __name__ == '__main__':

    set_font()
    draw_figures()

