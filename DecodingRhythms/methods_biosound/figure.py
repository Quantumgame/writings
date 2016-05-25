import os
import operator

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from lasp.signal import power_spectrum
from lasp.sound import temporal_envelope, spec_colormap, plot_spectrogram, log_transform

from lasp.timefreq import gaussian_stft

from neosound.sound_store import HDF5Store
from neosound.sound_manager import SoundManager
from DecodingRhythms.utils import set_font, get_this_dir

from zeebeez.aggregators.biosound import AggregateBiosounds
from zeebeez.utils import ALL_ACOUSTIC_PROPS, ACOUSTIC_PROP_COLORS_BY_TYPE


def get_syllable_props(agg, stim_id, syllable_order, data_dir):

    wave_pad = 20e-3
    i = (agg.df.stim_id == stim_id) & (agg.df.syllable_order == syllable_order)
    bird = agg.df[i].bird.values[0]
    start_time = agg.df[i].start_time.values[0] - wave_pad
    end_time = agg.df[i].end_time.values[0] + wave_pad
    xindex = agg.df[i].xindex.values[0]
    aprops = {aprop: agg.Xraw[xindex, k] for k, aprop in enumerate(ALL_ACOUSTIC_PROPS)}
    aprops['start_time'] = start_time
    aprops['end_time'] = end_time

    duration = end_time - start_time
    sfile = os.path.join(data_dir, bird, 'stims.h5')

    # get the raw sound pressure waveform
    sound_manager = SoundManager(HDF5Store, sfile, read_only=True, db_args={'read_only': True})
    wave = sound_manager.reconstruct(stim_id)
    wave_sr = wave.samplerate
    wave = np.array(wave).squeeze()
    wave_t = np.arange(len(wave)) / wave_sr
    wave_si = np.min(np.where(wave_t >= start_time)[0])
    wave_ei = wave_si + int(duration * wave_sr)
    amp_env = temporal_envelope(wave, wave_sr, cutoff_freq=200.0)
    amp_env /= amp_env.max()
    amp_env *= wave.max()

    # compute the spectrogram
    spec_sr = 1000.
    spec_t, spec_freq, spec, spec_rms = gaussian_stft(wave, float(wave_sr), 0.007, 1. / spec_sr, min_freq=300.,
                                                      max_freq=8000.)
    spec = np.abs(spec) ** 2
    log_transform(spec, dbnoise=70)
    spec_si = np.min(np.where(spec_t >= start_time)[0])
    spec_ei = spec_si + int(duration * spec_sr)

    # compute power spectrum
    ps_freq, ps = power_spectrum(wave[wave_si:wave_ei], wave_sr, log=False, hanning=False)

    return {'wave_t':wave_t, 'wave':wave, 'wave_si':wave_si, 'wave_ei':wave_ei, 
            'ps_freq':ps_freq, 'ps':ps, 'amp_env':amp_env,
            'spec_t':spec_t, 'spec_freq':spec_freq, 'spec':spec, 'spec_si':spec_si, 'spec_ei':spec_ei,
            'aprops':aprops
            }


def get_syllable_examples(agg, aprop, data_dir, bird=None):
    
    if bird is not None:
        i = agg.df.bird == bird
        df = agg.df[i]
    else:
        df = agg.df

    aprop_index = list(agg.acoustic_props).index(aprop)
    dlist = list()
    for stype,stim_id,syllable_order,xindex in zip(df.stim_type.values, df.stim_id.values, df.syllable_order.values, df.xindex.values):
        if stype in ['song', 'Ag', 'Di', 'mlnoise']:
            continue
        dlist.append( (stype, stim_id, syllable_order, agg.Xraw[xindex, aprop_index]))

    dlist.sort(key=operator.itemgetter(-1))
    print dlist

    med_index = int(len(dlist) / 2.)

    slist = list()
    slist.append(dlist[2])
    slist.append(dlist[med_index])
    slist.append(dlist[-1])

    specs = list()
    spec_freq = None
    stypes = list()
    propvals = list()
    for stype,stim_id,syllable_order,propval in slist:
        sprops = get_syllable_props(agg, stim_id, syllable_order, data_dir)
        s = sprops['spec']
        spec_freq = sprops['spec_freq']
        the_spec = s[:, sprops['spec_si']:sprops['spec_ei']]
        specs.append(the_spec)
        stypes.append(stype)
        propvals.append(propval)

    spec_lens = [s.shape[1] for s in specs]
    spec_len = np.sum(spec_lens)
    spec_space = 100

    full_spec_len = spec_space*(len(slist)+1) + spec_len
    full_spec = np.zeros([len(spec_freq), full_spec_len])

    centers = list()

    last_i = 0
    for k,s in enumerate(specs):
        si = last_i + spec_space
        ei = si + s.shape[1]
        c = int(((ei-si) / 2.) + si)
        centers.append(c)
        last_i = ei
        full_spec[:, si:ei] = s

    return spec_freq,full_spec,centers,propvals,stypes


def plot_syllable_comps(agg, stim_id=43, syllable_order=1, data_dir='/auto/tdrive/mschachter/data'):

    sprops = get_syllable_props(agg, stim_id, syllable_order, data_dir)
    wave = sprops['wave']
    wave_t = sprops['wave_t']
    wave_si = sprops['wave_si']
    wave_ei = sprops['wave_ei']
    ps_freq = sprops['ps_freq']
    ps = sprops['ps']
    amp_env = sprops['amp_env']
    aprops = sprops['aprops']
    start_time = aprops['start_time']
    spec = sprops['spec']
    spec_t = sprops['spec_t']
    spec_freq = sprops['spec_freq']
    spec_si = sprops['spec_si']
    spec_ei = sprops['spec_ei']

    aprop_specs = dict()
    aprop_spec_props = ['entropytime', 'meanspect', 'sal']
    for aprop in aprop_spec_props:
        aprop_specs[aprop] = get_syllable_examples(agg, aprop, data_dir, bird='GreBlu9508M')

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize, facecolor='w')
    fig.subplots_adjust(top=0.95, bottom=0.08, right=0.97, left=0.06, hspace=0.30, wspace=0.30)
    gs = plt.GridSpec(3, 100)

    sp_width = 20

    ax = plt.subplot(gs[0, :sp_width])
    plt.plot((wave_t[wave_si:wave_ei] - start_time)*1e3, wave[wave_si:wave_ei], 'k-', linewidth=2.)
    plt.plot((wave_t[wave_si:wave_ei] - start_time)*1e3, amp_env[wave_si:wave_ei], 'r-', linewidth=4., alpha=0.7)
    meantime = aprops['meantime']*1e3
    stdtime = aprops['stdtime']*1e3
    plt.axvline(meantime, color='r', linestyle='--', linewidth=3.0, alpha=0.9)
    plt.axvline(meantime-stdtime, color='r', linestyle='--', linewidth=3.0, alpha=0.8)
    plt.axvline(meantime+stdtime, color='r', linestyle='--', linewidth=3.0, alpha=0.8)
    plt.xlabel('Time (ms)')
    plt.ylabel('Waveform')
    plt.axis('tight')

    aprops_to_get = ['meantime', 'stdtime', 'skewtime', 'kurtosistime', 'entropytime', 'maxAmp']
    units = ['s', 's', '', '', '', '']
    ax = plt.subplot(gs[0, sp_width:(sp_width+18)])
    txt_space = 0.1
    for k,aprop in enumerate(aprops_to_get):
        aval = aprops[aprop]
        if aval > 10:
            txt = '%s: %d' % (aprop, aval)
        else:
            txt = '%s: %0.2f' % (aprop, aval)
        txt += ' %s' % units[k]
        plt.text(0.1, 1-((k+1)*txt_space), txt, fontsize=18)
    ax.set_axis_off()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(gs[1, :sp_width])
    fi = (ps_freq > 0) & (ps_freq <= 8000.)
    plt.plot(ps_freq[fi]*1e-3, ps[fi], 'k-', linewidth=3., alpha=1.)
    for aprop in ['q1', 'q2', 'q3']:
        plt.axvline(aprops[aprop]*1e-3, color='#606060', linestyle='--', linewidth=3.0, alpha=0.9)
        # plt.text(aprops[aprop]*1e-3 - 0.6, 3000., aprop, fontsize=14)
    plt.ylabel("Power")
    plt.xlabel('Frequency (kHz)')
    plt.axis('tight')

    aprops_to_get = ['meanspect', 'stdspect', 'skewspect', 'kurtosisspect', 'entropyspect', 'q1', 'q2', 'q3']
    units = ['Hz', 'Hz', '', '', '', 'Hz', 'Hz', 'Hz']
    ax = plt.subplot(gs[1, sp_width:(sp_width+18)])
    for k,aprop in enumerate(aprops_to_get):
        aval = aprops[aprop]
        if aval > 10:
            txt = '%s: %d' % (aprop, aval)
        else:
            txt = '%s: %0.2f' % (aprop, aval)
        txt += ' %s' % units[k]
        plt.text(0.1, 1-((k+1)*txt_space), txt, fontsize=18)
    ax.set_axis_off()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    spec_colormap()
    ax = plt.subplot(gs[2, :sp_width])
    plot_spectrogram((spec_t[spec_si:spec_ei]-start_time)*1e3, spec_freq*1e-3, spec[:, spec_si:spec_ei], ax, colormap='SpectroColorMap', colorbar=False)
    plt.axhline(aprops['fund']*1e-3, color='k', linestyle='--', linewidth=3.0)
    plt.ylabel("Frequency (kHz")
    plt.xlabel('Time (ms)')

    for k,aprop in enumerate(aprop_spec_props):

        full_spec_freq, full_spec, centers, propvals, stypes = aprop_specs[aprop]

        ax = plt.subplot(gs[k, (sp_width+20):])
        full_spec_t = np.arange(full_spec.shape[1])
        plot_spectrogram(full_spec_t, full_spec_freq*1e-3, full_spec, ax, colormap='SpectroColorMap', colorbar=False)
        for c,st in zip(centers,stypes):
            plt.text(c, 7., st, fontsize=20)
        pstrs = list()
        for p in propvals:
            if p > 10:
                pstrs.append('%d' % p)
            else:
                pstrs.append('%0.3f' % p)
        plt.xticks(centers, pstrs)
        plt.xlabel(aprop)

    aprops_to_get = ['fund', 'fund2', 'sal', 'voice2percent', 'maxfund', 'minfund', 'cvfund']
    units = ['Hz', 'Hz', '', '', 'Hz', 'Hz', 'Hz']
    ax = plt.subplot(gs[2, sp_width:(sp_width + 18)])
    for k, aprop in enumerate(aprops_to_get):
        aval = aprops[aprop]
        if aval > 10:
            txt = '%s: %d' % (aprop, aval)
        else:
            txt = '%s: %0.2f' % (aprop, aval)
        txt += ' %s' % units[k]
        plt.text(0.1, 1 - ((k + 1) * txt_space), txt, fontsize=18)
    ax.set_axis_off()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([])
    plt.yticks([])

    fname = os.path.join(get_this_dir(), 'figure.svg')
    # plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def plot_acoustic_stats(agg, data_dir='/auto/tdrive/mschachter/data'):

    Xz,good_indices = agg.remove_duplicates()
    i = np.zeros(len(agg.df), dtype='bool')
    i[good_indices] = True
    aprops = agg.acoustic_props

    df = agg.df[i]
    dur = (df.end_time - df.start_time).values * 1e3
    dur_q1 = np.percentile(dur, 1)
    dur_q5 = np.percentile(dur, 5)
    dur_q25 = np.percentile(dur, 25)
    dur_q50 = np.percentile(dur, 50)

    dur_thresh = 45
    i = (dur > dur_thresh) & (dur < 500)

    print '# of samples: %d' % i.sum()

    # compute the correlation matrix
    C = np.corrcoef(Xz.T)

    # build an undirected graph from the correlation matrix
    g = nx.Graph()
    for aprop in aprops:
        rgb = ACOUSTIC_PROP_COLORS_BY_TYPE[aprop]
        viz = {'color':{'r':rgb[0], 'g':rgb[1], 'b':rgb[2], 'a':0}}
        g.add_node(aprop, name=str(aprop), viz=viz)

    # connect nodes if their cc is above a given threshold
    cc_thresh = 0.25
    for k,aprop1 in enumerate(aprops):
        for j in range(k):
            if np.abs(C[k, j]) > cc_thresh:
                aprop2 = aprops[j]
                viz = {'color': {'r': 0, 'g': .5, 'b': 0, 'a': 0.7}}
                g.add_edge(aprop1, aprop2, weight=abs(float(C[k, j])))

    # pos = nx.spring_layout(g)
    # nx.draw(g, pos, labels={aprop:aprop for aprop in aprops})
    # plt.show()

    nx.write_gexf(g, '/tmp/acoustic_feature_graph.gexf')

    print 'Acoustic Feature Clusters:'
    for grp in sorted(nx.connected_components(g), key=len, reverse=True):
        print grp

    figsize = (23, 12)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.20, hspace=0.01, wspace=0.01)
    gs = plt.GridSpec(1, 100)

    # plot data matrix
    absmax = np.abs(Xz).max()
    absmax = 4
    ax = plt.subplot(gs[0, :40])
    plt.imshow(Xz, interpolation='nearest', aspect='auto', vmin=-absmax, vmax=absmax, cmap=plt.cm.seismic)
    xt = range(len(aprops))
    plt.xticks(xt, agg.acoustic_props, rotation=90)
    plt.colorbar()
    plt.title('Z-scored Acoustic Feature Matrix')

    ax = plt.subplot(gs[0, 50:])
    plt.imshow(C, interpolation='nearest', aspect='auto', vmin=-1, vmax=1, cmap=plt.cm.seismic, origin='lower')
    xt = range(len(aprops))
    plt.xticks(xt, aprops, rotation=90)
    plt.yticks(xt, aprops)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Acoustic Feature Correlation Matrix')

    fname = os.path.join(get_this_dir(), 'acoustic_data_and_corr_matrix.svg')
    plt.savefig(fname, facecolor='w', edgecolor='none')

    plt.show()


def plot_meantime_stuff(agg):

    durs = agg.df.end_time - agg.df.start_time

    i = (durs > 0.040) & (durs < 0.400)
    aprops = list(agg.acoustic_props)
    mti = aprops.index('meantime')
    sti = aprops.index('stdtime')
    xi = agg.df.xindex[i].values

    meantime = agg.Xraw[xi, mti]
    stdtime = agg.Xraw[xi, sti]
    durs = durs[i]

    plt.figure()
    ax = plt.subplot(1, 2, 1)
    plt.plot(meantime, durs, 'ko')
    plt.xlabel('Meantime')
    plt.ylabel('Duration')

    ax = plt.subplot(1, 2, 2)
    plt.plot(stdtime, durs, 'ko')
    plt.xlabel('Stdtime')
    plt.ylabel('Duration')

    plt.show()


if __name__ == '__main__':

    set_font()

    agg_file = '/auto/tdrive/mschachter/data/aggregate/biosound.h5'
    agg = AggregateBiosounds.load(agg_file)

    # plot_syllable_comps(agg)
    # plot_acoustic_stats(agg)
    plot_meantime_stuff(agg)

