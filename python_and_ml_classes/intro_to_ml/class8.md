1. Square Matrices and Eigenvectors
2. Covariance Matrices
3. Principal Components Analysis
4. Eigenfaces


===1. Square Matrices and Eigenvectors=== 

**Eigenvalues** and **eigenvectors** are useful and generic descriptions of what a given matrix is like. They describe the matrix's personality. Any square NxN matrix has N eigenvalues and N eigenvectors. Every eigenvalue of a matrix A is associated with an eigenvector, and each eigen value/vector pair satisfies this equation:

[[math]]
A\textbf{v} ~=~ \lambda \textbf{v}
[[math]]

**When we multiply A by an eigenvector, we get a scalar multiple of that eigenvector**. This winds up being super important in all sorts of ways that we encounter throughout our mathematical lifetimes. We'll see that they are used in Principal Components Analysis, for example.

[[@http://en.wikipedia.org/wiki/Normal_matrix|Normal matrices]] (also called invertible or non-singular) have a set of distinct and orthogonal eigenvalues/vectors, while non-normal (singular) matrices may have eigenvalues that repeat and eigenvectors that are not orthogonal to each other.

Eigenvectors and values can be [[@http://en.wikipedia.org/wiki/Complex_number|complex-valued]]. Complex numbers are like two dimensional vectors, and they live in a two-dimensional space called the **complex plane**. Also in the complex plane there is the special extra number i:

[[math]]
i ~=~ \sqrt{-1}
[[math]]

Any complex number z can be written as:

[[math]]
z = x + iy
[[math]]

or represented as a two dimensional vector:

[[math]]
z =
\left[ \begin{array}{c}
x \\
y
\end{array}
\right]
[[math]]

x is the **real** part and y is the **imaginary** part. Complex numbers can also be written in polar form:

[[math]]
z = r e^{i\theta}
[[math]]

r is the **amplitude** and theta is the **phase**. The elements of a vector can be complex-valued. An n-dimensional complex vector **z** would be written like this:

[[math]]
\textbf{z} \in \mathbb{C}^N
[[math]]

Is this blowing your mind? You should [[@https://www.khanacademy.org/math/algebra2/complex-numbers-a2/complex_numbers/v/introduction-to-complex-numbers|learn some more]] about the complex number system. All we'll use for our purposes is the two-dimensional representation of a complex number so that we can **plot the eigenvalues** of a matrix. That way some day when you're dealing with large square matrices and wondering how to characterize them, you'll know just what to do!

Something to know before we get going is that a normal matrix A can be decomposed into a matrix of eigenvectors and a diagonal matrix of eigenvalues, like this:

[[math]]
A ~=~ V U V^{-1}
[[math]]

Each column of V is an eigenvector, and U is a diagonal matrix of eigenvalues. The nonzero entry in the first column of U is the eigenvalue that corresponds to the first eigenvector.

As an exercise we're going to construct a 50x50 Gaussian random matrix. It will have 50 eigenvalues and 50 eigenvectors. We'll compute them by using NumPy's [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html|eig]] function, and plot the eigenvalues on the complex plane:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

def eigenvector_example(N):

    #construct a Gaussian random matrix
    A = np.random.randn(N, N)
    #rescale it so it's maximum value is one
    absmax = np.abs(A).max()
    A /= absmax

    #compute the eigenvalues and eigenvectors of A
    eigenvalues,eigenvectors = np.linalg.eig(A)

    #we want to plot the complex-valued eigenvalues.  we'll
    #consider their real part the x coordinate and the imaginary
    #part the y coordinate.

    plt.figure()
    #first plot the random matrix
    plt.subplot(2, 1, 1)
    plt.imshow(A, interpolation='nearest', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Random Matrix')

    #then plot the eigenvalue spectrum
    plt.subplot(2, 1, 2)
    #plot the unit circle
    phase = np.linspace(-np.pi, np.pi, 200)
    xcirc = np.cos(phase)
    ycirc = np.sin(phase)
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.plot(xcirc, ycirc, 'k-')
    plt.plot(eigenvalues.real, eigenvalues.imag, 'ro')
    plt.axis('tight')
    plt.title('Eigenvalues')

eigenvector_example(50)
plt.show()
[[code]]


===2. Covariance Matrices=== 

You may recall from a few classes ago that the [[@http://en.wikipedia.org/wiki/Covariance|covariance]] is a measure of how two random variables change together. Well, now we're dealing with many random variables, such as the features in our regressions. We might have hundreds of pixels in an image, for example. Think about images of handwritten digits - a handwritten zero has many pixels that **covary** with each other. That covariance is essential to what makes a zero different from a one.

A [[@http://en.wikipedia.org/wiki/Covariance_matrix|covariance matrix]] is simply a matrix of covariances - element i,j of the matrix is the covariance between feature i and feature j. The covariance matrix is [[@http://en.wikipedia.org/wiki/Symmetric_matrix|symmetric]], entry i,j is equal to entry j,i. The numpy function [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html|cov]] can be used to compute a covariance matrix, given a transposed data matrix. Let's compute the covariance matrix for handwritten ones and zeros:

[[code format="python"]]
import matplotlib.cm as cm
from sklearn.datasets import *
def plot_digit_covariance(digit=0):

    #load handwritten digits for 0
    data_dict = load_digits()

    #get the target values
    y = data_dict['target']

    #get the feature matrix to regress on
    X = data_dict['data']

    #select out the digits of interest
    index = y == digit

    #compute the covariance matrix of the zeros
    C = np.cov(X[index, :].T)

    #compute the covariance of a random matrix that is the same size
    R = np.random.randn(64, index.sum())
    Crand = np.cov(R)

    #plot the covariance matrix
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(C, interpolation='nearest', aspect='auto', cmap=cm.afmhot, vmin=0)
    plt.colorbar()
    plt.title('Covariance matrix for {}'.format(digit))

    plt.subplot(2, 1, 2)
    plt.imshow(Crand, interpolation='nearest', aspect='auto', cmap=cm.afmhot, vmin=0)
    plt.colorbar()
    plt.title('Covariance for random matrix')

plot_digit_covariance(0)
plt.show()
[[code]]

NumPy implements the **empirical estimator** for the covariance matrix. Estimating a covariance matrix is ultimately a maximum likelihood problem, and similar to regression, the more samples you have, the better the estimation of the covariance matrix. Scikits provides some more sophisticated [[@http://scikit-learn.org/stable/modules/covariance.html|covariance estimators]] that you could use when you are data limited.


===3. Principal Components Analysis=== 

Any group of random variables that have nonzero covariance with each other exhibit [[@http://en.wikipedia.org/wiki/Redundancy_%28information_theory%29|redundancy]]. In the case of handwritten digits, if a few pixels are turned off, you can still perceive a zero, and so can an algorithm. Any group of feature vectors that are redundant can be **compressed** into smaller sized feature vectors. The process of compressing a large feature vector into a smaller feature vector is called [[@http://en.wikipedia.org/wiki/Dimensionality_reduction|dimensionality reduction]], and is a big part of Machine Learning.

[[@http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf|Principal Components Analysis]] is a method for dimensionally reducing data. It does that in a specific way - it uses a matrix of **principal components (PCs)** to **project** a high dimensional feature vector into a low dimensional feature vector. The PCs are found in a way such that the variance of the first component's projection has the highest variance, the second has the second highest, and so forth. The projections are made such that **as much variance in the data as possible is captured by the top PCs**.

How does PCA do this? For a derivation, see section 12.1.1 of PRML. Start with the first component. We want to find a principal component vector that we can multiply against our data matrix to produce a lower-dimensional vector. Another way of saying that is that we want to **project** the data **onto** the **principal component**. We want the variance of that projection to be as high as possible.

It turns out mathematically that **the principal components** **are the eigenvectors of the covariance matrix**. The PCs are not complex-valued, because **symmetric matrices have real-valued eigenvalues and eigenvectors**. That's why we talked about eigenvectors earlier. If you wanted to build your own PCA algorithm, all you have to do is this:

# Z-score your data so each feature has zero mean and a variance of 1.
# Compute the covariance matrix of your data.
# Compute the eigenvalues and eigenvectors of the covariance matrix. The eigenvectors are the principal components.

There are other ways of deriving and obtaining the principal components. For example, if you have so many features that it's not feasible to compute a covariance matrix, and you only want a few PCs, try [[@http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.RandomizedPCA.html|randomized PCA]]. And more generally, there are [[@http://scikit-learn.org/stable/modules/decomposition.html|many decomposition methods]] out there that you can play with.

Let's build some intuition about PCA using a very common example. We will generate 2D data from a [[@http://en.wikipedia.org/wiki/Multivariate_normal_distribution|Multivariate Gaussian]], run PCA on it, look at the PCs, and look at the PC projections of the data.

[[code format="python"]]
from sklearn.decomposition import PCA
def pca_example(N=1000):
    """ In this example we'll illustrate PCA in two dimensions using data
        generated from a multivariate Gaussian.
    """

    #construct a 2x2 covariance matrix
    C = np.array([[1.0, 0.5], [0.5, 1.0]])

    #generate samples from a 2D multivariate Gaussian
    X = np.random.multivariate_normal(np.array([0.0, 0.0]), C, size=N)

    #fit PCA on the data
    pca = PCA()
    pca.fit(X)

    #project the data onto the principal components
    Xproj = pca.transform(X)

    #print the covariance matrices of raw and projected data
    Craw = np.cov(X.T)
    Cproj = np.cov(Xproj.T)
    print 'Raw data covariance matrix:'
    print Craw
    print 'Projected data covariance matrix:'
    print Cproj

    plt.figure()

    #plot the raw data
    plt.subplot(2, 1, 1)
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.plot(X[:, 0], X[:, 1], 'ko')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Raw Data')

    #plot the PCs over the raw data
    pc1 = pca.components_[0, :]*3
    pc2 = pca.components_[1, :]*3

    plt.plot([0.0, pc1[0]], [0.0, pc1[1]], 'r-', linewidth=3.0)
    plt.plot([0.0, pc2[0]], [0.0, pc2[1]], 'g-', linewidth=3.0)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.subplot(2, 1, 2)
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.plot(Xproj[:, 0], Xproj[:, 1], 'ro')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.title('Projected Data')

pca_example(1000)
[[code]]

Some important things to note about this example:
# The PCs align with the directions **in the raw data that** exhibit the most variance**.**
# The PCs are **orthogonal to each other.**
# The **projections are independent from each other**, the covariance matrix of the projections is **diagonal**.


===4. Eigenfaces=== 

An [[@http://en.wikipedia.org/wiki/Eigenface|Eigenface]] is a principal component derived from a set of images of faces. They look cool. Here we'll use PCA on a set of face images to demonstrate image compression.

This example will be ever-so-slightly more involved than previous ones. First, make a directory for the code, and then make a subdirectory called "images". Download this file and extract the images into the "images" subdirectory:

[[file:faces.zip]]

Then put this code in a script, save it to the code directory (the parent directory of "images") and run it:

[[code format="python"]]
def plot_eigenfaces(num_pcs=36):
    import glob
    #get the 64x64 grayscale image names from the current directory
    image_names = glob.glob("images/*.jpg")

    #construct a data matrix of flattened images
    X = list()
    for iname in image_names:
        #read the image from the file
        img = plt.imread(iname)
        X.append(img.ravel())
    X = np.array(X)

    assert num_pcs <= X.shape[0], "There are only {} data points, can't have more PCs than that!".format(X.shape[0])

    #do PCA on the matrix of images
    pca = PCA(n_components=num_pcs)
    pca.fit(X)

    #project the data into a lower dimensional subspace
    Xproj = pca.transform(X)

    #plot some of the actual faces
    plt.figure()
    plt.suptitle('Actual Faces')
    nrows = 6
    ncols = 6
    num_plots = nrows*ncols
    for k,flat_img in enumerate(X[:num_plots, :]):
        #reshape the flattened image into a matrix
        img = flat_img.reshape([64, 64])
        plt.subplot(nrows, ncols, k)
        plt.imshow(img, interpolation='nearest', aspect='auto')

    #plot some of the eigenfaces and the variance they capture
    plt.figure()
    plt.suptitle('Eigenfaces')
    nrows = int(np.ceil(np.sqrt(num_pcs)))
    ncols = nrows
    num_plots = min(nrows*ncols, num_pcs)
    for k,explained_variance in enumerate(pca.explained_variance_ratio_[:num_plots]):
        #get the principal component
        pc = pca.components_[k, :]

        #reshape it into an image
        pc_img = pc.reshape([64, 64])

        #plot the principle component
        plt.subplot(nrows, ncols, k)
        plt.imshow(pc_img, interpolation='nearest', aspect='auto')

        #show the variance captured by this component
        plt.title('EV: {:.3f}'.format(explained_variance))

    #reconstruct the compressed images
    nrows = 6
    ncols = 6
    num_plots = nrows*ncols
    plt.figure()
    plt.suptitle('Reconstructed Images')
    for k,compressed_img in enumerate(Xproj[:num_plots, :]):
        #uncompress the image
        flat_img = pca.inverse_transform(compressed_img)
        img = flat_img.reshape([64, 64])
        plt.subplot(nrows, ncols, k)
        plt.imshow(img, interpolation='nearest', aspect='auto')

plot_eigenfaces(num_pcs=36)
plt.show()
[[code]]

Unfortunately for this example, we only have 46 images, and cannot have more PCs than samples! But I think it's good enough for you to get the gist of what's going on. We can compress a 64x64 image, which has 4096 features, into a 36 dimensional vector and still preserve much of the information!
