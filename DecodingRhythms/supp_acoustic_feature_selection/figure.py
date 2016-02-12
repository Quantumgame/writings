import os

import numpy as np
import matplotlib.pyplot as plt

from zeebeez.transforms.biosound import BiosoundTransform


def plot_acoustic_stats(bird='GreBlu9508M', data_dir='/auto/tdrive/mschachter/data'):

    trans_dir = os.path.join(data_dir, bird, 'transforms')
    bfile = os.path.join(trans_dir, 'BiosoundTransform_%s.h5' % bird)

    bst = BiosoundTransform.load(bfile)

    df = bst.stim_df
    dur = (df.end_time - df.start_time).values * 1e3
    dur_q1 = np.percentile(dur, 1)
    dur_q5 = np.percentile(dur, 5)
    dur_q25 = np.percentile(dur, 25)
    dur_q50 = np.percentile(dur, 50)

    dur_thresh = 45
    i = (dur > dur_thresh) & (dur < 500)

    print '# of samples: %d' % i.sum()

    reduced_aprops = ['fund',
                       'sal',
                       'voice2percent',
                       'meanspect',
                       'skewspect',
                       'entropyspect',
                       'q2',
                       'meantime',
                       'skewtime',
                       'entropytime',
                       'maxAmp']

    # build a matrix of acoustic properties
    naprops = len(bst.acoustic_props)
    X = np.zeros([i.sum(), naprops])
    for k,aprop in enumerate(bst.acoustic_props):
        X[:, k] = df[i][aprop].values

    # build a matrix of acoustic properties
    Xred = np.zeros([i.sum(), len(reduced_aprops)])
    for k,aprop in enumerate(reduced_aprops):
        Xred[:, k] = df[i][aprop].values

    # compute the correlation matrix
    C = np.corrcoef(X.T)
    Cred = np.corrcoef(Xred.T)

    fig = plt.figure()
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.09, hspace=0.01, wspace=0.01)
    gs = plt.GridSpec(1, 1)

    # ax = plt.subplot(gs[0, 0])
    # plt.hist(dur[i], bins=20, color='g', alpha=0.6)
    # plt.xlabel('Duration (ms)')
    # plt.title('Syllable Durations')
    # plt.axis('tight')

    # plot full correlation matrix
    ax = plt.subplot(gs[0, 0])
    plt.imshow(C, interpolation='nearest', aspect='auto', vmin=-1, vmax=1, cmap=plt.cm.seismic)
    xt = range(naprops)
    plt.xticks(xt, bst.acoustic_props, rotation=45)
    plt.yticks(xt, bst.acoustic_props)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Acoustic Feature Correlation Matrix')

    # plot reduced correlation matrix
    fig = plt.figure()
    fig.subplots_adjust(left=0.07, right=0.99, top=0.98, bottom=0.09, hspace=0.01, wspace=0.01)
    gs = plt.GridSpec(1, 1)

    ax = plt.subplot(gs[0, 0])
    plt.imshow(Cred, interpolation='nearest', aspect='auto', vmin=-1, vmax=1, cmap=plt.cm.seismic)
    xt = range(len(reduced_aprops))
    plt.xticks(xt, reduced_aprops, rotation=45)
    plt.yticks(xt, reduced_aprops)
    plt.colorbar(label='Correlation Coefficient')
    plt.title('Acoustic Feature Correlation Matrix')


if __name__ == '__main__':

    plot_acoustic_stats()
    plt.show()

