import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from DecodingRhythms.utils import set_font, log_transform, get_this_dir
from lasp.sound import spec_colormap, plot_spectrogram
from lasp.timefreq import power_spectrum_jn
from zeebeez.transforms.rnn_preprocess import RNNPreprocessTransform
from zeebeez.utils import CALL_TYPE_COLORS


def get_env_psds(rp):
    # compute the stim amp envelope
    stim_env = rp.U.sum(axis=1)
    stim_env /= stim_env.max()

    # grab the amplitude envelope for each stimulus
    print 'Computing psds...'
    stim_ids = rp.event_df.stim_id.unique()
    env_data = {'stim_id': list(), 'stim_type': list(), 'duration': list(), 'xindex': list()}
    env_psds = list()
    env_freq = None
    duration_thresh = 1.1
    for stim_id in stim_ids:
        i = rp.event_df.stim_id == stim_id
        stim_type = rp.event_df[i].stim_type.values[0]
        stime = rp.event_df[i].start_time.values[0]
        etime = rp.event_df[i].end_time.values[0]
        dur = etime - stime
        if dur < duration_thresh:
            print 'Stim %d (%s) is too short: %0.3fs' % (stim_id, stim_type, dur)
            continue
        si = int(stime * rp.sample_rate)
        ei = int(etime * rp.sample_rate)

        u = stim_env[si:ei]

        env_freq, env_ps, env_ps_var, env_phase = power_spectrum_jn(u, rp.sample_rate, window_length=1.0,
                                                                    increment=0.200)

        env_data['stim_id'].append(stim_id)
        env_data['stim_type'].append(stim_type)
        env_data['duration'].append(etime - stime)
        env_data['xindex'].append(len(env_psds))

        env_psds.append(env_ps)

    env_psds = np.array(env_psds)
    fi = (env_freq >= 2.) & (env_freq <= 30.)

    env_psds = env_psds[:, fi]
    env_freq = env_freq[fi]
    env_df = pd.DataFrame(env_data)

    return env_freq,env_psds,env_df


def plot_specs_with_env(ax, rp, stim_id, trial=3):

    # plot distance call w/ amp envelope
    i = (rp.event_df.stim_id == stim_id) & (rp.event_df.trial == trial)
    assert i.sum() == 1

    stime = rp.event_df[i].start_time.values[0]
    etime = rp.event_df[i].end_time.values[0]
    si = int(stime*rp.sample_rate)
    ei = int(etime * rp.sample_rate)

    spec = rp.U[si:ei, :].T
    spec_freq = rp.spec_freq
    spec_env = spec.sum(axis=0)
    spec_env /= spec_env.max()
    spec_t = np.arange(spec.shape[1]) / rp.sample_rate

    spec_env *= (spec_freq.max() - spec_freq.min())
    spec_env += spec_freq.min()
    plot_spectrogram(spec_t, spec_freq, spec, ax=ax, ticks=True, fmax=8000., colormap='SpectroColorMap', colorbar=False)
    plt.plot(spec_t, spec_env, 'k-', linewidth=5.0, alpha=0.7)
    plt.axis('tight')


def plot_temp_mods(ax, rp):

    env_freq, env_psds, env_df = get_env_psds(rp)

    log_transform(env_psds)

    affiliative_calls = (env_df.stim_type == 'DC') | (env_df.stim_type == 'Te') | (env_df.stim_type == 'LT')
    song_calls = (env_df.stim_type == 'song')
    noise_calls = (env_df.stim_type == 'mlnoise')
    beg_calls = (env_df.stim_type == 'Be')

    calls = [affiliative_calls, song_calls, beg_calls, noise_calls]
    psds = list()
    for call_i in calls:
        xi = env_df.xindex[call_i].values
        _psds = env_psds[xi, :]
        psd_mean = _psds.mean(axis=0)
        psd_std = _psds.std(axis=0, ddof=1)
        psds.append([psd_mean, psd_std])

    fi = env_freq < 20.

    plt.sca(ax)
    clrs = [CALL_TYPE_COLORS['DC'], 'k', 'g', 'r']
    for k, (psd, psd_std) in enumerate(psds):
        plt.plot(env_freq[fi], psd[fi], '-', linewidth=8.0, alpha=0.7, c=clrs[k])
    plt.axis('tight')
    plt.legend(['Affiliative', 'Song', 'Begging', 'ML-Noise'])
    plt.xlabel('Temporal Modulation Frequency (Hz)')
    plt.ylabel('Power (dB)')


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):

    preproc_file = os.path.join(data_dir, 'GreBlu9508M', 'preprocess', 'RNNPreprocess_GreBlu9508M_Site4_Call1_L.h5')
    rp = RNNPreprocessTransform.load(preproc_file)

    i = rp.event_df.stim_type == 'mlnoise'
    print 'ml noise stims: ',rp.event_df[i].stim_id.unique()

    fig = plt.figure(figsize=(23, 10), facecolor='w')
    fig.subplots_adjust(hspace=0.35, wspace=0.35, right=0.95, left=0.10)
    gs = plt.GridSpec(3, 2)

    ax = plt.subplot(gs[0, 0])
    plot_specs_with_env(ax, rp, 287)

    ax = plt.subplot(gs[1, 0])
    plot_specs_with_env(ax, rp, 277)

    ax = plt.subplot(gs[2, 0])
    plot_specs_with_env(ax, rp, 68)

    ax = plt.subplot(gs[:, 1])
    plot_temp_mods(ax, rp)

    fname = os.path.join(get_this_dir(), 'figure.svg')
    plt.savefig(fname, facecolor=fig.get_facecolor(), edgecolor='none')

    plt.show()


if __name__ == '__main__':
    set_font()
    spec_colormap()
    draw_figures()