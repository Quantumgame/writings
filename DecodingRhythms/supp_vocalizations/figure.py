import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

from lasp.sound import spectrogram, plot_spectrogram, spec_colormap

from zeebeez.core.experiment import Experiment


def draw_figures(data_dir='/auto/tdrive/mschachter/data', bird='GreBlu9508M', output_dir='/auto/tdrive/mschachter/data/sounds'):

    spec_colormap()

    exp_dir = os.path.join(data_dir, bird)
    exp_file = os.path.join(exp_dir, '%s.h5' % bird)
    stim_file = os.path.join(exp_dir, 'stims.h5')
    
    exp = Experiment.load(exp_file, stim_file)
    
    bird = exp.bird_name
    all_stim_ids = list()
    # iterate through the segments and get the stim ids from each epoch table
    for seg in exp.get_all_segments():
        etable = exp.get_epoch_table(seg)
        stim_ids = etable['id'].unique()
        all_stim_ids.extend(stim_ids)

    stim_ids = np.unique(all_stim_ids)

    stim_info = list()
    for stim_id in stim_ids:
        si = exp.stim_table['id'] == stim_id
        assert si.sum() == 1, "More than one stimulus defined for id=%d" % stim_id
        stype = exp.stim_table['type'][si].values[0]
        if stype == 'call':
            stype = exp.stim_table['callid'][si].values[0]

        # get sound pressure waveform
        sound = exp.sound_manager.reconstruct(stim_id)
        waveform = np.array(sound.squeeze())
        sample_rate = float(sound.samplerate)
        stim_dur = len(waveform) / sample_rate

        stim_info.append( (stim_id, stype, sample_rate, waveform, stim_dur))

    durations = np.array([x[-1] for x in stim_info])
    max_dur = durations.max()
    min_dur = durations.min()

    max_fig_size = 15.
    min_fig_size = 5.

    for stim_id,stype,sample_rate,waveform,stim_dur in stim_info:

        fname = os.path.join(output_dir, '%s_stim_%d.wav' % (stype, stim_id))
        print 'Writing %s...' % fname
        wavfile.write(fname, sample_rate, waveform)

        dfrac = (stim_dur - min_dur) / (max_dur - min_dur)
        fig_width = dfrac*(max_fig_size - min_fig_size) + min_fig_size

        spec_t,spec_freq,spec,rms = spectrogram(waveform, sample_rate, sample_rate, 136., min_freq=300, max_freq=8000,
                                                log=True, noise_level_db=80, rectify=True, cmplx=False)

        figsize = (fig_width, 5)
        fig = plt.figure(figsize=figsize)
        plot_spectrogram(spec_t, spec_freq, spec, colormap='SpectroColorMap', colorbar=False)
        plt.title('Stim %d: %s' % (stim_id, stype))

        fname = os.path.join(output_dir, '%s_stim_%d.png' % (stype, stim_id))
        plt.savefig(fname, facecolor='w')
        plt.close('all')


if __name__ == '__main__':

    draw_figures()
