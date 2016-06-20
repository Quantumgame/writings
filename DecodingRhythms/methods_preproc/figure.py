import os

import matplotlib.pyplot as plt
import numpy as np

from DecodingRhythms.utils import set_font, get_this_dir
from lasp.colormaps import viridis
from lasp.sound import plot_spectrogram, spec_colormap
from DecodingRhythms.utils import get_full_data
from zeebeez.utils import USED_ACOUSTIC_PROPS, ACOUSTIC_PROP_COLORS_BY_TYPE, ACOUSTIC_PROP_NAMES


def plot_full_data(d, syllable_index):

    syllable_start = d['syllable_props'][syllable_index]['start_time'] - 0.020
    syllable_end = d['syllable_props'][syllable_index]['end_time'] + 0.030

    figsize = (24.0, 10)
    fig = plt.figure(figsize=figsize, facecolor='w')
    fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20, wspace=0.20)

    gs = plt.GridSpec(100, 100)
    left_width = 55
    top_height = 30
    middle_height = 40
    # bottom_height = 40
    top_bottom_sep = 20

    # plot the spectrogram
    ax = plt.subplot(gs[:top_height+1, :left_width])
    spec = d['spec']
    spec[spec < np.percentile(spec, 15)] = 0
    plot_spectrogram(d['spec_t'], d['spec_freq']*1e-3, spec, ax=ax, colormap='SpectroColorMap', colorbar=False, ticks=True)
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.ylabel('Frequency (kHz)')
    plt.xlabel('Time (s)')

    # plot the LFPs
    sr = d['lfp_sample_rate']
    # lfp_mean = d['lfp'].mean(axis=0)
    lfp_mean = d['lfp'][2, :, :]
    lfp_t = np.arange(lfp_mean.shape[1]) / sr
    nelectrodes,nt = lfp_mean.shape
    gs_i = top_height + top_bottom_sep
    gs_e = gs_i + middle_height + 1

    ax = plt.subplot(gs[gs_i:gs_e, :left_width])

    voffset = 5
    for n in range(nelectrodes):
        plt.plot(lfp_t, lfp_mean[nelectrodes-n-1, :] + voffset*n, 'k-', linewidth=3.0, alpha=0.75)
    plt.axis('tight')
    ytick_locs = np.arange(nelectrodes) * voffset
    plt.yticks(ytick_locs, list(reversed(d['electrode_order'])))
    plt.ylabel('Electrode')
    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.xlabel('Time (s)')

    # plot the PSTH
    """
    gs_i = gs_e + 5
    gs_e = gs_i + bottom_height + 1
    ax = plt.subplot(gs[gs_i:gs_e, :left_width])
    ncells = d['psth'].shape[0]
    plt.imshow(d['psth'], interpolation='nearest', aspect='auto', origin='upper', extent=(0, lfp_t.max(), ncells, 0),
               cmap=psth_colormap(noise_level=0.1))

    cell_i2e = d['cell_index2electrode']
    print 'cell_i2e=',cell_i2e
    last_electrode = cell_i2e[0]
    for k,e in enumerate(cell_i2e):
        if e != last_electrode:
            plt.axhline(k, c='k', alpha=0.5)
            last_electrode = e

    ytick_locs = list()
    for e in d['electrode_order']:
        elocs = np.array([k for k,el in enumerate(cell_i2e) if el == e])
        emean = elocs.mean()
        ytick_locs.append(emean+0.5)
    plt.yticks(ytick_locs, d['electrode_order'])
    plt.ylabel('Electrode')

    plt.axvline(syllable_start, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    plt.axvline(syllable_end, c='k', linestyle='--', linewidth=3.0, alpha=0.7)
    """

    # plot the biosound properties
    sprops = d['syllable_props'][syllable_index]
    aprops = USED_ACOUSTIC_PROPS

    vals = [sprops[a] for a in aprops]
    ax = plt.subplot(gs[:top_height, (left_width+5):])
    plt.axhline(0, c='k')
    for k,(aprop,v) in enumerate(zip(aprops,vals)):
        bx = k
        rgb = np.array(ACOUSTIC_PROP_COLORS_BY_TYPE[aprop]).astype('int')
        clr_hex = '#%s' % "".join(map(chr, rgb)).encode('hex')
        plt.bar(bx, v, color=clr_hex, alpha=0.7)

    # plt.bar(range(len(aprops)), vals, color='#c0c0c0')
    plt.axis('tight')
    plt.ylim(-1.5, 1.5)
    plt.xticks(np.arange(len(aprops))+0.5, [ACOUSTIC_PROP_NAMES[aprop] for aprop in aprops], rotation=90)
    plt.ylabel('Z-score')

    # plot the LFP power spectra
    gs_i = top_height + top_bottom_sep
    gs_e = gs_i + middle_height + 1

    f = d['psd_freq']
    ax = plt.subplot(gs[gs_i:gs_e, (left_width+5):])
    plt.imshow(sprops['lfp_psd'], interpolation='nearest', aspect='auto', origin='upper',
               extent=(f.min(), f.max(), nelectrodes, 0), cmap=viridis, vmin=-2., vmax=2.)
    plt.colorbar(label='Z-scored Log Power')
    plt.xlabel('Frequency (Hz)')
    plt.yticks(np.arange(nelectrodes)+0.5, d['electrode_order'])
    plt.ylabel('Electrode')

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def draw_figures():

    d = get_full_data('GreBlu9508M', 'Site4', 'Call1', 'L', 268)
    # d = get_full_data('GreBlu9508M', 'Site4', 'Call1', 'L', 277)
    plot_full_data(d, 1)

    plt.show()


if __name__ == '__main__':
    spec_colormap()
    set_font()
    draw_figures()
