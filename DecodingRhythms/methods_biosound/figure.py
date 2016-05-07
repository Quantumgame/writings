import os

import numpy as np
import matplotlib.pyplot as plt

from lasp.signal import power_spectrum
from lasp.sound import temporal_envelope, spec_colormap, plot_spectrogram, log_transform

from lasp.timefreq import gaussian_stft

from neosound.sound_store import HDF5Store
from neosound.sound_manager import SoundManager
from utils import set_font

from zeebeez.aggregators.biosound import AggregateBiosounds
from zeebeez.utils import ALL_ACOUSTIC_PROPS


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
    sound_manager = SoundManager(HDF5Store, sfile, db_args={'read_only': True})
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

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(top=0.95, bottom=0.08, right=0.97, left=0.06, hspace=0.20, wspace=0.20)
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

    ax = plt.subplot(gs[1, :sp_width])
    fi = (ps_freq > 0) & (ps_freq <= 8000.)
    plt.plot(ps_freq[fi]*1e-3, ps[fi], 'k-', linewidth=3., alpha=1.)
    for aprop in ['q1', 'q2', 'q3']:
        plt.axvline(aprops[aprop]*1e-3, color='#606060', linestyle='--', linewidth=3.0, alpha=0.9)
        # plt.text(aprops[aprop]*1e-3 - 0.6, 3000., aprop, fontsize=14)
    plt.ylabel("Power")
    plt.xlabel('Frequency (kHz)')
    plt.axis('tight')

    spec_colormap()
    ax = plt.subplot(gs[2, :sp_width])
    plot_spectrogram((spec_t[spec_si:spec_ei]-start_time)*1e3, spec_freq*1e-3, spec[:, spec_si:spec_ei], ax, colormap='SpectroColorMap', colorbar=False)
    plt.axhline(aprops['fund']*1e-3, color='k', linestyle='--', linewidth=3.0)
    plt.ylabel("Frequency (kHz")
    plt.xlabel('Time (ms)')

    plt.show()


if __name__ == '__main__':

    set_font()

    agg_file = '/auto/tdrive/mschachter/data/aggregate/biosound.h5'
    agg = AggregateBiosounds.load(agg_file)

    plot_syllable_comps(agg)