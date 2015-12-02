import os

from zeebeez.transforms.biosound import BiosoundTransform
from zeebeez.transforms.pairwise_cf import PairwiseCFTransform
from zeebeez.transforms.stim_event import StimEventTransform


def get_full_data(bird, block, segment, hemi, stim_id, data_dir='/auto/tdrive/mschachter/data'):

    bdir = os.path.join(data_dir, bird)
    tdir = os.path.join(bdir, 'transforms')

    # load the BioSound
    bs_file = os.path.join(bdir, 'BiosoundTransform_%s.h5' % bird)
    bs = BiosoundTransform.load(bs_file)

    # load the StimEvent transform

    se_file = os.path.join(tdir, 'StimEvent_%s_%s_%s_%s.h5' % (bird,block,segment,hemi))
    print 'Loading %s...' % se_file
    se = StimEventTransform.load(se_file, rep_types_to_load=['raw'])
    se.segment_stims_from_biosound(bs_file)

    # load the pairwise CF transform
    pcf_file = os.path.join(tdir, 'PairwiseCF_%s_%s_%s_%s_raw.h5' % (bird,block,segment,hemi))
    print 'Loading %s...' % pcf_file
    pcf = PairwiseCFTransform.load(pcf_file)

    # get the spectrogram
    spec_freq = se.spec_freq
    stim_spec = se.spec_by_stim[stim_id]

    # get the raw LFP
    lfp = se.lfp_reps_by_stim['raw'][stim_id]
    ntrials,nelectrodes,nt = lfp.shape

    # get the raw spikes, spike_mat is ragged array of shape (num_trials, num_cells, num_spikes)
    spike_mat = se.spikes_by_stim[stim_id]
    assert ntrials == len(spike_mat)

    # get the syllable start and end times
    i = se.segment_df.stim_id == stim_id
    syllable_df = se.segment_df[i]

    # get the biosound properties
    bs.stim_df

    i = pcf.df.stim_id == stim_id
    df = pcf.df[i]









def draw_figures():

    pass


if __name__ == '__main__':
    draw_figures()