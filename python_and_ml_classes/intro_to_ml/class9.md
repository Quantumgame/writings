1. Unsupervised Data and Clustering
2. Multivariate Gaussians
3. K-Means Clustering
4. Gaussian Mixture Models
5. Graph Clustering

===1. Unsupervised Data and Clustering=== 

To **cluster** data means to take a bunch of data points, and assign a label to each point in some meaningful way, so that **points close to each other are assigned the same label**. When we are not given the labels, the problem is **unsupervised**, we can't use a regression to predict a category from a data point. There are two situations in which we can cluster data:
# We have a bunch of unlabelled feature vectors.
# All we have are the distances between data points, with no actual features.

For the first situation, we can use K-Means, Gaussian Mixture Models, and other algorithms. For the second situation, we need to use graph based algorithms.


===2. Multivariate Gaussians=== 

We would like to generate some 2D points as our data, corresponding to features in a two-dimensional feature space. The points will come from from one of several 2D [[@http://en.wikipedia.org/wiki/Multivariate_normal_distribution|Multivariate Gaussians]]. A Multivariate Gaussian is a [[@http://en.wikipedia.org/wiki/Multivariate_random_variable|random vector]] that can be completely described by a mean vector and a covariance matrix. As noted in an earlier class, we can use NumPy's [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multivariate_normal.html|multivariate_normal]] function to generate samples from a Multivariate Gaussian, as long as we specify the mean vector and covariance matrix. Here's some code that plots samples from three different ones:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

def generate_gaussmix(num_samples_per_cluster=100, plot=True, cluster_probs=(0.33, 0.33, 0.33)):
    """ Generate data from a mixture of 2D Gaussians. """

    #the centers of each Gaussian distribution
    centers = [[1.0, 1.0], [-2.5, -2], [-1.5, 2]]

    #specify the covariance matrix for each Gaussian
    cmats = ([[1.0, 0.3], [0.3, 1.0]],
             [[1.0, 0.75], [0.75, 1.0]],
             [[1.0, 0.0], [0.0, 1.0]])

    #generate random samples for each distribution
    X = list()
    y = list()
    for k,(mean,cov_mat) in enumerate(zip(centers, cmats)):
        nsamps = int(num_samples_per_cluster*cluster_probs[k])
        X.extend(np.random.multivariate_normal(mean, cov_mat, size=nsamps))
        y.extend([k]*nsamps)

    X = np.array(X)
    y = np.array(y)

    if plot:
        clusters = np.unique(y)
        plt.figure()
        for k in clusters:
            plt.plot(X[y == k, 0], X[y == k, 1], 'o')
        plt.title('Data From 3 MV Gaussians')

    return X,y

X,y = generate_gaussmix(num_samples_per_cluster=100, plot=True)
plt.show()
[[code]]


===3. KMeans Clustering=== 

[[@http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans|KMeans]] clustering is the most straightforward, and perhaps primitive, way to cluster data. It can scale to large datasets and serves as a good baseline clustering algorithm to compare other algorithms against.

There are some good [[@https://www.youtube.com/watch?v=0MQEt10e4NM|videos]] out there on KMeans that describe how it works. The basic strategy is this:

# Choose a number of clusters.
# Pick a random mean for each cluster.
# Assign a label to each datapoint according to the mean it is closest to.
# Recompute the means for each cluster.
# Repeat step 3.

The mean of each cluster is also called a **centroid**. Let's use KMeans to cluster our Gaussian distributions:

[[code format="python"]]
from sklearn.cluster import KMeans

def run_kmeans(X, num_clusters=3, plot=True):

    km = KMeans(n_clusters=num_clusters)
    km.fit(X)
    ypred = km.predict(X)

    if plot:
        plt.figure()
        for k in range(num_clusters):
            plt.plot(X[ypred == k, 0], X[ypred == k, 1], 'o')
        plt.title('KMeans Result, num_clusters=%d' % num_clusters)

    return ypred

X,y = generate_gaussmix(num_samples_per_cluster=100, plot=True)
ypred = run_kmeans(X, num_clusters=3, plot=True)
plt.show()
[[code]]


===4. Gaussian Mixture Models=== 

The problem with KMeans is that it assumes that each cluster can be completely described by it's mean. [[@http://scikit-learn.org/stable/modules/mixture.html|Gaussian Mixture Models]], on the other hand, explicitly fit not only the means and covariances of each cluster, but also the probability of each cluster occurring.

When it comes to unsupervised clustering, all we are given are the data points, also called the **observed variables**. We use our imagination or intuition to determine that the data points are grouped together into different clusters. The cluster that a data point belongs to is a [[@http://en.wikipedia.org/wiki/Latent_variable|latent variable]], a discrete random variable that we believe exists but have not observed.

It's hard to maximize the likelihood of a model with latent variables, we need to use [[@http://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm|expectation maximization]] to do it. We won't go over EM, but the basic idea of it is this:

# Initialize the EM algorithm with random guesses for the means, covariances, and cluster probabilities.
# Estimate the most likely cluster for each data point given the current guess for the means, covariances, and cluster probabilities.
# Find a new parameter by maximizing the **complete log likelihood**, which is the likelihood function if we assume the latent variables are observed.
# Repeat step 2 and 3 until convergence of the parameters.

Do you notice the similarities between the algorithm for KMeans and the EM algorithm for GMMs? KMeans is in fact a special type of GMM where the covariance matrices are assumed to be diagonal.

Let's try to fit our data with a Gaussian Mixture Model:

[[code format="python"]]
from sklearn.mixture import GMM

def run_gmm(X, num_clusters=3, plot=True):

    gmm = GMM(n_components=num_clusters, covariance_type='full')
    gmm.fit(X)
    ypred = gmm.predict(X)

    #print out the information for each fit cluster
    for k in range(num_clusters):
        the_mean = gmm.means_[k]
        the_cov_mat = gmm.covars_[k]
        the_cluster_prob = gmm.weights_[k]
        print 'Cluster %d' % k
        print '\tProbability: %0.2f' % the_cluster_prob
        print '\tMean: ',the_mean
        print '\tCovariance:'
        print the_cov_mat

    if plot:
        plt.figure()
        for k in range(num_clusters):
            plt.plot(X[ypred == k, 0], X[ypred == k, 1], 'o')
        plt.title('GMM Clustering, num_clusters=%d' % num_clusters)

    return ypred

X,y = generate_guassmix(num_samples_per_cluster=100, plot=True, cluster_probs=[0.15, 0.35, 0.5])
ypred = run_gmm(X, num_clusters=3)
[[code]]

**EXERCISE**: Fit more and less than 3 clusters to the data. Does it look reasonable? If you didn't know that there were three clusters, given the number of data points do you think you would still be confident that there are three clusters? Since we know the ground truth for this problem (comparing y to ypred), use a clustering [[@http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation|performance metric]] to determine what the best number of clusters are that fit the data.


===5. Graph Clustering=== 

Some data cannot be broken down into feature vectors. For example, imagine Twitter hires you to analyze the data of their social network. All you really have is the interactions between users of that network - retweets, tweeting at eachother, and so forth. Social networks can be modelled as [[@http://en.wikipedia.org/wiki/Graph_%28mathematics%29|graphs]], which are comprised of **nodes** and **edges**. The nodes are the people, the edges represent the interactions between people.

Graphs can be **directed** or **undirected**, depending on whether the edges are directional or non-directional. In the Twitter example, the edge between two people could be directed, and indicate the number of times a person tweeted at another person. There would be two directed edges between those two people. For our examples here, we are going to use undirected graphs, because directed graphs are harder to cluster.

First, before clustering, let's use the package [[@https://networkx.github.io/|NetworkX]] to construct a plot a graph:

[[code format="python"]]
import networkx as nx
import matplotlib.cm as cm

def generate_social_graph(num_nodes=100, num_clusters=4, plot=True):

    g = nx.Graph()

    #create all the nodes first
    for k in range(num_nodes):
        #randomly select a cluster
        c = np.random.randint(num_clusters)

        #construct nodes
        g.add_node(k, cluster=c)

    #now connect the nodes with undirected edges, nodes
    #in the same cluster will have a stronger weight
    #than those outside of their cluster
    for n1 in range(num_nodes):
        #get the cluster for node 1
        c1 = g.node[n1]['cluster']

        for n2 in range(n1):
            #get the cluster for node 2
            c2 = g.node[n2]['cluster']

            #determine the edge weight
            if c1 == c2:
                #when nodes are in the same cluster, create a
                #strong nonzero connection between them
                w = np.random.rand()*3
            else:
                #when nodes are in different clusters, create
                #a weak connection between them with 10% probabilitys
                w = np.random.rand()
                if w < 0.9:
                    w = 0.0
            #create the edge in the graph
            if w > 0:
                g.add_edge(n1, n2, weight=w)

    if plot:
        plt.figure()
        cluster_colors = ['r', 'g', 'b', 'y']
        node_clrs = [cluster_colors[g.node[n]['cluster']] for n in g.nodes()]
        pos = nx.spectral_layout(g, scale=1)
        weights = [g[n1][n2]['weight'] for n1,n2 in g.edges()]

        nx.draw_networkx(g, pos=pos, node_color=node_clrs, edge_cmap=cm.Greys, edge_vmin=0.0, edge_vmax=3.0, edge_color=weights)

    return g

g = generate_social_graph(num_nodes=100, num_clusters=4, plot=True)
[[code]]

In the network generated above, nodes are clustered, so that some groups of nodes are more strongly connected to each other than others. We would like to find these clusters in an unsupervised way, given the weights between nodes. To do this, we use the [[@http://en.wikipedia.org/wiki/Adjacency_matrix|adjacency matrix]], which is a matrix of edge weights. Element i,j of an adjacency matrix is the weight between node i and node j. In an undirected graph, the adjacency matrix is symmetric.

A [[@http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html|spectral clustering]] from scikits can take the adjacency matrix and use it to cluster the graph. Spectral clustering, in a not-so-clear nutshell, does k-means clustering on the eigenvectors of the adjacency matrix. If that makes sense to you, well, then you don't even need to read the [[@http://en.wikipedia.org/wiki/Spectral_clustering|wiki page]] on the topic.

Anyways, here is some code that takes the constructed graph, clusters it, and then displays the elements of the cluster in text form:

[[code format="python"]]
from sklearn.cluster import SpectralClustering

def run_spectral_cluster(g, num_clusters=4):

    #generate an adjacency matrix, element ij is
    #the weight betweeen node i and node j
    W = np.asarray(nx.adjacency_matrix(g)).squeeze()

    #create a spectral clustering object
    sc = SpectralClustering(affinity='precomputed', n_clusters=num_clusters)

    #run spectral clustering on the adjacency matrix
    sc.fit(W)

    #return the clusters for each node
    return sc.labels_

num_clusters = 4
g = generate_social_graph(100, num_clusters=num_clusters, plot=True)
ypred = run_spectral_cluster(g, num_clusters=num_clusters)

#aggregate nodes by cluster
actual_clusters = {c:list() for c in range(num_clusters)}
predicted_clusters = {c:list() for c in range(num_clusters)}
for n in g.nodes():
    c = g.node[n]['cluster']
    cpred = ypred[n]
    actual_clusters[c].append(n)
    predicted_clusters[cpred].append(n)

#print out the actual clusters
for k in range(num_clusters):
    print 'Actual Cluster %d:' % k
    print actual_clusters[k]
print ''

#print out the predicted clusters
for k in range(num_clusters):
    print 'Predicted Cluster %d:' % k
    print predicted_clusters[k]

plt.show()
[[code]]

If you'd like to learn more about clustering, check out the papers listed on [[@http://www.cs.berkeley.edu/~jordan/courses/294-fall09/lectures/clustering/|this lecture's webpage]].
