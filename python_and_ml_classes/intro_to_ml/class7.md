1. Working with Image Data
2. Logistic Regression
3. Confusion Matrices and Performance Metrics
4. Nonlinear Decision Boundaries and Support Vector Machines

===1. Working with Image Data=== 

For our example we're going to use some sample [[@http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits|image data]] from scikits. Images come in the form of matrices. In order to run a machine learning algorithm on an image, we consider each pixel a separate feature, and flatten the matrix of pixels into a vector of pixels. Let's load the scikits data and plot it:

[[code format="python"]]
import numpy as np
from sklearn.datasets import *

[[code]]
import matplotlib.cm as cm
[[code format="python"]]
import matplotlib.pyplot as plt

def load_binary_digits(noise_std=5.00):
    """
        Loads up some handwritten digit data from scikits and plots it.

        noise_std: The standard deviation of the guassian noise
            added to the image to corrupt it.
    """

    #load handwritten digits for 0 and 1
    data_dict = load_digits(n_class=2)

    #get the binary output targets
    y = data_dict['target']

    #get the feature matrix to regress on
    X = data_dict['data']

    #each pixel takes on a gray level value from 1-16 that indicates
    #intensity. we're going to treat the data as if it was continuous
    #and z-score it.

    #subtract off the mean for each feature
    X -= X.mean(axis=0)
    #compute the standard deviation for each feature
    Xstd = X.std(axis=0, ddof=1)
    #divide the feature matrix by the nonzero stds of features
    nz = Xstd > 0.0
    X[:, nz] /= Xstd[nz]

    #add gaussian noise to the images
    X += np.random.randn(X.shape[0], X.shape[1])*noise_std

    #plot some of the images
    nrows = 10
    ncols = 10
    nplots = nrows*ncols
    plt.figure()
    for k in range(nplots):
        #reshape the features into an image
        img = X[k].reshape([8, 8])
        plt.subplot(nrows, ncols, k+1)
        plt.imshow(img, interpolation='nearest', aspect='auto', cmap=cm.gist_yarg)
        plt.xticks([])
        plt.yticks([])

    return X,y

X,y = load_binary_digits()
plt.show()
[[code]]


===2. Logistic Regression=== 

When we do supervised learning to predict a category, we call it **classification**. Nevertheless, history has insisted on calling the **linear** model that predicts a **binary** category **logistic regression**. Remember that linear regression tries to predict a continuous variable from a linear combination of features:

[[math]]
\hat{y} ~=~ w_1 x_1 ~+~ \cdots ~+~ w_M x_M ~+~ b
[[math]]

We //could// just make the assumption that our 0, 1 values represent real numbers and do linear regression on the data, but it turns out there's a better way. What we do instead is **map** the output of a linear model to a number between 0 and 1, using a [[@http://en.wikipedia.org/wiki/Logistic_function|logistic function]] (also called **sigmoid**). Here's the equation for it:

[[math]]
\sigma(x) ~=~ \left( 1 + e^{-x} \right) ^{-1}
[[math]]

Let's make a plot of it:

[[code format="python"]]
x = np.linspace(-7, 7, 100)
plt.axhline(0.5, color='b')
plt.axvline(0.0, color='r')
plt.plot(x, 1.0 / (1.0 + np.exp(-x)), 'k-', linewidth=2.0)
plt.axis('tight')
plt.ylabel('Probability of 1')
plt.xlabel('Linear Output')
plt.show()
[[code]]

Some things to note about the logistic function:
# It is bounded between 0 and 1.
# When the linear model predicts a 0, the probability of a 1 is exactly 0.5. Feature vectors where the linear prediction is 0 lie on the [[@http://en.wikipedia.org/wiki/Decision_boundary|decision boundary]] of the model, where the output could be either a 0 or a 1.

The logistic function is not arbitrary, it can be derived from a probabilistic interpretation of the problem (see Chapter 4.2 of PRML). The cost function that we need to minimize to find the optimal weights can be derived through maximum likelihood. If we assume our output variable is [[@http://en.wikipedia.org/wiki/Bernoulli_distribution|Bernoulli]] distributed, and it's parameter p is equal to the sigmoid applied to a linear combination of features like this:

[[math]]
\hat{p} (\textbf{w}, b) = \sigma \left( w_1 x_1 ~+~ \cdots ~+~ w_M x_M ~+~ b \right)
[[math]]

then the likelihood function looks like this:

[[math]]
P (y_1, \cdots, y_N ~|~ \textbf{w}, b) ~=~ \prod_{i=1}^N ~ \hat{p}_i^{y_i} ~\left( 1 - \hat{p}_i \right)^{1 - y_i}
[[math]]

and the negative log likelihood is called the [[@http://en.wikipedia.org/wiki/Cross_entropy|cross entropy]] function:

[[math]]
E(\textbf{w}, b) ~= -\sum_{i=1}^N \left(~ y_i log( \hat{p}_i ) ~+~ (1 - y_i) log ( 1 - \hat{p}_i ) ~ \right)
[[math]]

So be aware of the following things:
# In linear regression, the optimal parameters are the ones that maximize the likelihood, which is the same thing as the minimum of the negative log likelihood. The negative log likelihood in this case is the sum-of-squares cost function.
# In logistic regression, we also maximize the likelihood by minimizing the negative log likelihood, and the negative log likelihood is the cross entropy cost function.


===3. Confusion Matrices and Performance Metrics=== 

Remember that the feature vectors where the linear part of the logistic regression model predicts a zero lie on the **decision boundary**. The linear part of the logistic regression prediction is called the **score**. When the score is negative, the model is on the side of the decision boundary that predicts a zero, because when passed through the sigmoid we get probability for one of less than 0.5. When the score is positive, the prediction is a one because positive values passed through the sigmoid give a probability for one of more than 0.5.

When predicting a binary value, there are four different possibilities:

# **True Positive:** the target is a 1 and the prediction is a 1
# **False Negative (Type I Error):** The target is a 1 and the prediction is a 0
# **True Negative**: the target is a 0 and the prediction is a 0
# **False Positive (Type II Error)**: the target is a 0 and the prediction is a 1

For a given test set, we accumulate the counts of each type of prediction in a [[@http://en.wikipedia.org/wiki/Confusion_matrix|confusion matrix]]. For binary classification, the confusion matrix is a 2x2 matrix that looks like this:

[[math]]
\left[ \begin{array}{cc}
\text{# of TN} & \text{# of FP} \\
\text{# of FN} & \text{# of TP}
\end{array} \right]
[[math]]

When we divide the elements in the top row by the number of total ones in the test data, we get the true positive and false positive **rates**. Same goes for the bottom row when we divide by the number of total zeros in the test data. The rates are probabilities, and each row is a conditional probability distribution that sums to 1:

[[math]]
\left[ \begin{array}{cc}
P(~ \hat{p} < 0.5 ~|~ y = 0 ~) & P(~ \hat{p} > 0.5 ~|~ y = 0 ~) \\
P(~ \hat{p} < 0.5 ~|~ y = 1 ~) & P(~ \hat{p} > 0.5 ~|~ y = 1 ~) \\
\end{array} \right]
[[math]]

The average value of the diagonal is called the **percent correct**, and is often used as a performance metric. Another commonly used performance metric is the [[@http://en.wikipedia.org/wiki/Receiver_operating_characteristic|area under a ROC curve]], which we will not explore in detail. There are other metrics based on the rates, such as the [[@http://en.wikipedia.org/wiki/Sensitivity_and_specificity|sensitivity and specificity]], that you should check out.

Let's run a scikits [[@http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html|logistic regression]] on our image data and check out the confusion matrix. Now that we are mature machine learnists we will use the full framework of cross validation to find optimal regularization parameters.

[[code format="python"]]
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.datasets import *
from sklearn.cross_validation import KFold

import matplotlib.cm as cm
import matplotlib.pyplot as plt

def fit_binary_data(X, y, C_to_try=[1e-3, 1e-2, 1e-1, 1.0], nfolds=10):
    """
        Use logistic regression to fit the image data using cross validation.
    """

    best_C = None
    best_pc = 0
    best_weights = None
    best_intercept = None

    for C in C_to_try:

        aucs = list()
        pcs = list()
        weights = list()
        intercepts = list()
        cmats = list()

        for train_indices,test_indices in KFold(len(y), n_folds=nfolds):
            assert len(np.intersect1d(train_indices, test_indices)) == 0
            #break the data matrix up into training and test sets
            Xtrain, Xtest, ytrain, ytest = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

            #construct a logistic regression object
            lc = LogisticRegression(C=C)
            lc.fit(Xtrain, ytrain)

            #predict the identity of images on the test set
            ypred = lc.predict(Xtest)

            #compute confusion matrix
            cmat = confusion_matrix(ytest, ypred, labels=[0, 1]).astype('float')

            #normalize each row of the confusion matrix so they represent probabilities
            cmat = (cmat.T / cmat.sum(axis=1)).T

            #compute the percent correct
            pcs.append((cmat[0, 0] + cmat[1, 1]) / 2.0)

            #record the confusion matrix for this fold
            cmats.append(cmat)

            #predict the probability of a 1 for each test sample
            ytest_prob = lc.predict_proba(Xtest)[:, 1]

            #compute and record the area under the curve for the predictions
            auc = roc_auc_score(ytest, ytest_prob)
            aucs.append(auc)

            #record the weights and intercept
            weights.append(lc.coef_)
            intercepts.append(lc.intercept_)

        #compute the mean confusion matrix
        cmats = np.array(cmats)
        Cmean = cmats.mean(axis=0)

        #compute the mean AUC and PC
        mean_auc = np.mean(auc)
        std_auc = np.std(auc, ddof=1)
        mean_pc = np.mean(pcs)
        std_pc = np.std(pcs, ddof=1)

        #compute the mean weights
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)
        mean_intercept = np.mean(intercepts)

        print 'C={:.4f}'.format(C)
        print '\tPercent Correct: {:.3f} +/- {:.3f}'.format(mean_pc, std_pc)
        print '\tAUC: {:.3f} +/- {:.3f}'.format(mean_auc, std_auc)
        print '\tConfusion Matrix:'
        print Cmean

        #determine if we've found the best model thus far
        if mean_pc > best_pc:
            best_pc = mean_pc
            best_C = C
            best_weights = mean_weights
            best_intercept = mean_intercept

    #reshape the weights into an image
    weights_img = best_weights.reshape([8, 8])

    #make a plot of the weights
    weights_absmax = np.abs(best_weights).max()
    plt.figure()
    plt.imshow(weights_img, interpolation='nearest', aspect='auto', cmap=cm.seismic, vmin=-weights_absmax, vmax=weights_absmax)
    plt.colorbar()
    plt.title('Model Weights C: {:.4f}, PC: {:.2f}, Intercept: {:.3f}'.format(best_C, best_pc, best_intercept))
[[code]]

Call that function like this:

[[code format="python"]]
X,y = load_binary_digits(noise_std=5.0)
fit_binary_data(X, y)
plt.show()
[[code]]

**EXERCISE**: Run that code with varying amounts of noise and look how the percent correct and confusion matrices change.


===4. Nonlinear Decision Boundaries and Support Vector Machines=== 

Logistic Regression can only handle **linear decision boundaries**, which are planes in M-1 dimensional space that separate feature vectors. In practice feature vectors might be separated by **nonlinear decision boundaries**, which are not planes, but potentially something more complex. Here's an example of a nonlinear decision boundary:

[[code format="python"]]
def generate_nonlinear_data(plot=False):

    #the decision function is just some wacky arbitrary nonlinear function
    decision_function = lambda xp,yp: yp**3*np.cos(xp) + xp**3*np.sin(yp)

    npts = 25
    xvals = np.linspace(-2, 2, npts)
    yvals = np.linspace(-2, 2, npts)
    Xcoords,Ycoords = np.meshgrid(xvals, yvals)

    D = decision_function(Xcoords, Ycoords)

    #construct feature matrix and target vector from data
    X = list()
    y = list()
    for xval,yval,dval in zip(Xcoords.ravel(), Ycoords.ravel(), D.ravel()):
        #decide class based on sign of decision function
        clz = int(dval) > 0
        X.append( (xval, yval))
        y.append(clz)
    X = np.array(X)
    y = np.array(y)

    if plot:
        plt.figure()
        zi = y == 0
        plt.plot(X[zi, 0], X[zi, 1], 'ro')
        plt.plot(X[~zi, 0], X[~zi, 1], 'k^')
        plt.legend(['0', '1'])
        plt.axis('tight')

    return X,y

X,y = generate_nonlinear_data(plot=True)
plt.show()
[[code]]

Logistic Regression cannot handle this type of data, as this code will show:

[[code format="python"]]
def plot_2d_predictions(classifier, X, y):

    #construct a 2D grid of points and then turn it into a feature matrix called Xgrid
    npts = 25
    xvals = np.linspace(-2, 2, npts)
    yvals = np.linspace(-2, 2, npts)
    Xcoords,Ycoords = np.meshgrid(xvals, yvals)
    Xgrid = np.array(zip(Xcoords.ravel(), Ycoords.ravel()))

    #compute the predictions for each location in the 2D grid
    ypred = classifier.predict(Xgrid)

    #plot the actual data
    plt.figure()
    plt.subplot(2, 1, 1)
    zi = y == 0
    plt.plot(X[zi, 0], X[zi, 1], 'ro')
    plt.plot(X[~zi, 0], X[~zi, 1], 'k^')
    plt.legend(['0', '1'])
    plt.axis('tight')
    plt.title('Actual Data')

    #plot the predictions
    plt.subplot(2, 1, 2)
    zi = ypred == 0
    plt.plot(Xgrid[zi, 0], Xgrid[zi, 1], 'ro')
    plt.plot(Xgrid[~zi, 0], Xgrid[~zi, 1], 'k^')
    plt.legend(['0', '1'])
    plt.title('Predictions')
    plt.suptitle(classifier.__class__.__name__)

X,y = generate_nonlinear_data()
lc = LogisticRegression(C=1e-3)
lc.fit(X, y)
plot_2d_predictions(lc, X, y)
plt.show()
[[code]]

[[@http://en.wikipedia.org/wiki/Support_vector_machine|Support Vector Machines]] to the rescue! An SVM is another type of classifier. We won't go into too much detail about how they work, but here's some important things to know about them:

# They select a small subset of the feature vectors in your data that lie along the decision boundary, called **support vectors**, and use them for classification.
# They can implicitly project your low dimensional feature vectors into high dimensional space where a linear decision boundary exists, and use that boundary to do prediction.
# They can use a nonlinear **kernel** to do the projection and comparison between a feature vector to be predicted and the support vectors. You can specify the kernel.

Check out the [[@http://scikit-learn.org/stable/modules/svm.html|scikits support vector machine documentation]] to learn more. SVMs make short work of our nonlinear decision boundary example:

[[code format="python"]]
from sklearn.svm import SVC
svmc = SVC()
svmc.fit(X, y)
plot_2d_predictions(svmc, X, y)
plt.show()
[[code]]
