import os
import operator
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import KFold

from sklearn.linear_model import Ridge

from lasp.signal import bandpass_filter
from lasp.sound import plot_spectrogram, spec_colormap
from lasp.timefreq import power_spectrum_jn
from utils import get_full_data
from zeebeez.aggregators.biosound import AggregateBiosounds
from zeebeez.aggregators.tuning_curve import TuningCurveAggregator


def draw_tuning_curves(data_dir='/auto/tdrive/mschachter/data'):
    agg_file = os.path.join(data_dir, 'aggregate', 'tuning_curve.h5')
    agg = TuningCurveAggregator.load(agg_file)

    r2_thresh = 0.01
    i = (agg.df.decomp == 'full_psds') & (agg.df.freq == 99) & (agg.df.aprop == 'stdtime') & (agg.df.r2 > r2_thresh)
    assert i.sum() > 0

    xindex = agg.df.xindex[i]
    assert len(xindex) > 0

    lst = zip(xindex, agg.df.r2[i].values)
    lst.sort(key=operator.itemgetter(1))
    xindex = [x[0] for x in lst]

    cx = agg.curve_x[xindex, :]
    tc = agg.tuning_curves[xindex, :]

    plt.figure()
    for x, y in zip(cx, tc):
        plt.plot(x, y, 'k-', alpha=0.7)
    plt.xlabel('stdtime')
    plt.ylabel('Power (99Hz)')
    plt.xlim(0.02, 0.09)

    plt.show()


def draw_raw_lfp():

    spec_colormap()
    d = get_full_data('GreBlu9508M', 'Site4', 'Call1', 'L', 287)

    the_lfp = d['lfp'][2, :, :]

    nelectrodes,nt = the_lfp.shape
    bp_lfp = np.zeros_like(the_lfp)
    # bandpass from 95-105Hz
    for n in range(nelectrodes):
        bp_lfp[n, :] = bandpass_filter(the_lfp[n, :], d['lfp_sample_rate'], 95., 105.)

    bp_lfp = bp_lfp**2

    lfp_t = np.arange(nt) / d['lfp_sample_rate']

    figsize = (23, 13)
    fig = plt.figure(figsize=figsize)

    gs = plt.GridSpec(3, 1)

    ax = plt.subplot(gs[0, 0])
    plot_spectrogram(d['spec_t'], d['spec_freq'], d['spec'], ax=ax, colormap='SpectroColorMap', colorbar=True)

    ax = plt.subplot(gs[1, 0])
    absmax = np.abs(the_lfp).max()
    plt.imshow(the_lfp, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-absmax, vmax=absmax,
               extent=(lfp_t.min(), lfp_t.max(), 0, nelectrodes))
    plt.colorbar()

    ax = plt.subplot(gs[2, 0])
    absmax = np.abs(bp_lfp).max()
    plt.imshow(bp_lfp, interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot_r, vmin=0, vmax=absmax,
               extent=(lfp_t.min(), lfp_t.max(), 0, nelectrodes))
    plt.colorbar()

    plt.show()


def plot_meantime_stuff():

    agg_file = '/auto/tdrive/mschachter/data/aggregate/biosound.h5'
    agg = AggregateBiosounds.load(agg_file)
    durs = agg.df.end_time - agg.df.start_time

    i = (durs > 0.040) & (durs < 0.400)
    aprops = agg.acoustic_props
    mti = aprops.index('meantime')
    sti = aprops.index('stdtime')
    eti = aprops.index('entropytime')
    xi = agg.df.xindex[i].values

    meantime = agg.Xraw[xi, mti]
    stdtime = agg.Xraw[xi, sti]
    skewtime = agg.Xraw[xi,aprops.index('skewtime')]
    entropytime = agg.Xraw[xi, eti]
    durs = durs[i]

    plt.figure()
    ax = plt.subplot(2, 2, 1)
    plt.plot(meantime, durs, 'ko')
    plt.xlabel('Meantime')
    plt.ylabel('Duration')

    ax = plt.subplot(2, 2, 2)
    plt.plot(stdtime, durs, 'ko')
    plt.xlabel('Stdtime')
    plt.ylabel('Duration')
    plt.title('cc=%0.2f' % np.corrcoef(stdtime, durs)[0, 1])

    ax = plt.subplot(2, 2, 3)
    plt.plot(durs, entropytime, 'go')
    plt.ylabel('entropytime')
    plt.xlabel('duration')
    plt.title('cc=%0.2f' % np.corrcoef(entropytime, durs)[0, 1])

    ax = plt.subplot(2, 2, 4)
    plt.plot(durs, skewtime, 'go')
    plt.ylabel('skewtime')
    plt.xlabel('duration')
    plt.title('cc=%0.2f' % np.corrcoef(skewtime, durs)[0, 1])

    plt.show()


def draw_figures(data_dir='/auto/tdrive/mschachter/data'):
    # draw_tuning_curves()
    draw_raw_lfp()
    # plot_meantime_stuff()


def plot_random_r2_by_freq():

    nsamps = 1000

    dur_min = 0.070
    dur_max = 0.430
    sample_rate = 381.4697265625
    psd_window_size = 0.060
    psd_increment = 2 / sample_rate

    psds = list()
    psd_freq = None
    durations = np.random.rand(nsamps)*(dur_max - dur_min) + dur_min

    tstats = np.zeros([nsamps, 6])

    for k in range(nsamps):
        dur = durations[k]
        slen = int(dur*sample_rate)

        t = np.arange(slen) / sample_rate
        s = np.abs(np.random.randn(slen))
        # s = np.random.randn(slen)

        s_normed = np.abs(s) / np.abs(s).sum()
        meantime = np.dot(t, s_normed)
        vartime = np.dot((t-meantime)**2, s_normed)
        stdtime = np.sqrt(vartime)
        skewtime = np.dot( ((t-meantime) / stdtime)**3, s_normed)
        kurttime = np.dot( ((t-meantime) / stdtime)**4,  s_normed)
        nz = s_normed > 0
        entropytime = -np.sum(s_normed[nz]*np.log2(s_normed[nz]))

        tstats[k, :] = np.array([dur, meantime, stdtime, skewtime, kurttime, entropytime])

        psd_freq,psd,psd_phase = compute_psd(s, sample_rate, psd_window_size, psd_increment)
        psds.append(psd)
    psds = np.array(psds)

    X = deepcopy(psds)
    X -= X.mean(axis=0)
    X /= X.std(axis=0, ddof=1)

    plt.figure()
    plt.plot(durations, X[:, 0], 'ro')
    plt.xlabel('Duration')
    plt.ylabel('DC')
    plt.axis('tight')
    plt.show()

    tstat_names = ['duration', 'meantime', 'stdtime', 'skewtime', 'kurttime', 'entropytime']
    tstat_models = dict()
    for j,tname in enumerate(tstat_names):
        y = deepcopy(tstats[:, j])
        y -= y.mean()
        y /= y.std(ddof=1)

        tstat_models[tname] = bootstrap_fit(X, y)
        print 'Performance for %s: %0.2f' % (tname, tstat_models[tname]['r2'])

    plt.figure()
    nrows = 2
    ncols = 3
    for j,tname in enumerate(tstat_names):
        ax = plt.subplot(nrows, ncols, j+1)
        mdict = tstat_models[tname]
        plt.plot(psd_freq, mdict['W'], 'k-')
        plt.axis('tight')
        plt.ylim(0, 1)
        plt.title('%s: R2=%0.2f' % (tname, mdict['r2']))
    plt.show()


def compute_psd(s, sample_rate, window_length, increment):
    """ Computes the power spectrum of a signal. """

    min_freq = 0
    max_freq = sample_rate / 2

    freq, psd, psd_var, phase = power_spectrum_jn(s, sample_rate, window_length, increment, min_freq=min_freq,
                                                  max_freq=max_freq)

    # zero out frequencies where the lower bound dips below zero
    pstd = np.sqrt(psd_var)
    psd_lb = psd - pstd
    psd[psd_lb < 0] = 0

    return freq, psd, phase


def bootstrap_fit(X, y, nfolds=25, verbose=False):
    # set random seed to consistent value across runs, prior to forming training and validation sets
    np.random.seed(123456)

    nsamps = len(y)
    # construct training and validation sets using the bootstrap
    cv_indices = list(KFold(nsamps, n_folds=nfolds, shuffle=True))
    print 'len(cv_indices)=%d' % len(cv_indices)

    # keep track of the model performance for each set of model parameters
    model_perfs = list()

    # initialize a model class
    model_class = LinearModel
    hyperparams = model_class.get_hyperparams()

    # fit a distribution of models using bootstrap for each model parameter set in model_param_sets
    for hyperparam in hyperparams:

        if verbose:
            print 'Hyperparams:', hyperparam
        models = list()
        fold_data = {'ntrain': list(), 'ntest': list(), 'fold': list(),
                     'cc': list(), 'r2': list(), 'rmse': list(),
                     'likelihood': list(), 'likelihood_null': list()}

        # fit a model to each bootstrapped training set
        for k in range(nfolds):

            # grab the indices of training and validation sets
            train_indices,test_indices = cv_indices[k]
            assert len(np.intersect1d(train_indices, test_indices)) == 0, "Training and test sets overlap!"

            # get the training and validation matrices
            Xtrain = X[train_indices, :]
            ytrain = y[train_indices]

            Xtest = X[test_indices, :]
            ytest = y[test_indices]

            ntrain = len(ytrain)
            ntest = len(ytest)

            # construct a model and fit the data
            model = model_class(hyperparam)
            model.fit(Xtrain, ytrain)

            # make a prediction on the test set
            ypred = model.predict(Xtest)

            fold_data['ntrain'].append(ntrain)
            fold_data['ntest'].append(ntest)
            fold_data['fold'].append(k)

            d = compute_perfs(ytest, ypred, ytrain.mean())
            for key, val in d.items():
                fold_data[key].append(val)

            models.append(model)

            if verbose:
                print '\tFold %d: ntrain=%d, ntest=%d, cc=%0.2f, r2=%0.3f, rmse=%0.3f, likelihood=%0.3f' % \
                      (k, ntrain, ntest, d['cc'], d['r2'], d['rmse'], d['likelihood'])

        # compute the average performances
        fold_df = pd.DataFrame(fold_data)

        mean_r2 = fold_df['r2'].mean()
        stderr_r2 = fold_df['r2'].std(ddof=1) / np.sqrt(nfolds)

        # save the model info
        model_perfs.append({'hyperparam': hyperparam, 'models': models, 'fold_df': fold_df,
                            'mean_r2': mean_r2, 'stderr_r2': stderr_r2})

    # identify the best model parameter set, i.e. the one with the highest R2
    model_perfs.sort(key=operator.itemgetter('mean_r2'), reverse=True)

    best_model_dict = model_perfs[0]
    best_fold_df = best_model_dict['fold_df']
    best_hyperparam = best_model_dict['hyperparam']

    print 'Best Model Params:', best_model_dict['hyperparam']
    for pname in ['cc', 'r2', 'rmse', 'likelihood']:
        print '\t%s=%0.3f +/- %0.3f' % (pname, best_fold_df[pname].mean(), best_fold_df[pname].std(ddof=1))

    lk_full = best_fold_df.likelihood.values
    lk_null = best_fold_df.likelihood_null.values
    cv_r2 = best_fold_df.r2.mean()
    Wcv = np.array([m.get_weights() for m in best_model_dict['models']])
    W = Wcv.mean(axis=0)

    rdict = {'W': W, 'lk_full': lk_full, 'lk_null': lk_null, 'hyperparam': best_hyperparam, 'r2': cv_r2}

    return rdict


def compute_perfs(y, ypred, ymean):
    # compute correlation coefficient of prediction on test set
    cc = np.corrcoef(y, ypred)[0, 1]

    # compute r2 (likelihood ratio) of prediction on test set
    ss_res = np.sum((y - ypred) ** 2)
    ss_tot = np.sum((y - ymean) ** 2)
    r2 = 1. - (ss_res / ss_tot)

    # compute the RMSE
    rmse = np.sqrt(ss_res.mean())

    return {'cc': cc, 'r2': r2, 'rmse': rmse, 'likelihood': ss_res, 'likelihood_null': ss_tot}


class LinearModel(object):

    def __init__(self, lambda2):
        self.lambda2 = lambda2
        self.rr = Ridge(alpha=self.lambda2, fit_intercept=True)

    @classmethod
    def get_hyperparams(cls):
        return np.logspace(-2, 6, 50)

    def fit(self, X, y):
        self.rr.fit(X, y)

    def predict(self, X):
        return self.rr.predict(X)

    def get_weights(self):
        return self.rr.coef_


if __name__ == '__main__':
    # draw_figures()
    plot_random_r2_by_freq()
