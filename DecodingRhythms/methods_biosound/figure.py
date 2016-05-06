import os

import numpy as np
import matplotlib.pyplot as plt
from lasp.sound import temporal_envelope, spec_colormap, plot_spectrogram, log_transform

from lasp.timefreq import gaussian_stft

from neosound.sound_store import HDF5Store
from neosound.sound_manager import SoundManager

from zeebeez.aggregators.biosound import AggregateBiosounds


def plot_syllable_comps(agg, stim_id=43, syllable_order=1, data_dir='/auto/tdrive/mschachter/data'):

    wave_pad = 20e-3
    i = (agg.df.stim_id == stim_id) & (agg.df.syllable_order == syllable_order)
    bird = agg.df[i].bird.values[0]
    start_time = agg.df[i].start_time.values[0] - wave_pad
    end_time = agg.df[i].end_time.values[0] + wave_pad
    duration = end_time - start_time
    sfile = os.path.join(data_dir, bird, 'stims.h5')

    sound_manager = SoundManager(HDF5Store, sfile, db_args={'read_only':True})
    wave = sound_manager.reconstruct(stim_id)
    wave_sr = wave.samplerate
    wave = np.array(wave).squeeze()
    wave_t = np.arange(len(wave)) / wave_sr
    wave_si = np.min(np.where(wave_t >= start_time)[0])
    wave_ei = wave_si + int(duration*wave_sr)
    amp_env = temporal_envelope(wave, wave_sr, cutoff_freq=200.0)
    amp_env /= amp_env.max()
    amp_env *= wave.max()

    spec_sr = 1000.
    spec_t,spec_freq,spec,spec_rms = gaussian_stft(wave, float(wave_sr), 0.007, 1. / spec_sr, min_freq=300., max_freq=8000.)
    spec = np.abs(spec)**2
    log_transform(spec, dbnoise=70)
    spec_si = np.min(np.where(spec_t >= start_time)[0])
    spec_ei = spec_si + int(duration*spec_sr)

    figsize = (23, 13)
    plt.figure(figsize=figsize)
    gs = plt.GridSpec(3, 100)

    ax = plt.subplot(gs[0, :25])
    plt.plot((wave_t[wave_si:wave_ei] - start_time)*1e3, wave[wave_si:wave_ei], 'k-', linewidth=2.)
    plt.plot((wave_t[wave_si:wave_ei] - start_time)*1e3, amp_env[wave_si:wave_ei], 'r-', linewidth=4., alpha=0.7)
    plt.xlabel('Time (ms)')
    plt.ylabel('Waveform')
    plt.axis('tight')

    spec_colormap()
    ax = plt.subplot(gs[1, :25])
    plot_spectrogram((spec_t[spec_si:spec_ei]-start_time)*1e3, spec_freq*1e-3, spec[:, spec_si:spec_ei], ax, colormap='SpectroColorMap', colorbar=False)
    plt.ylabel("Frequency (kHz")
    plt.xlabel('Time (ms)')



    plt.show()


if __name__ == '__main__':

    agg_file = '/auto/tdrive/mschachter/data/aggregate/biosound.h5'
    agg = AggregateBiosounds.load(agg_file)

    plot_syllable_comps(agg)