===Outline=== 
1. The Joint and Conditional Probability between Two Random Variables
2. Independence, Covariance and Correlation
3. Conditional Probabilities
4. Fitting Data, Parameters, and Likelihood


===The Fake Data=== 

Sometimes if you want good data, you have to create it yourself! I mean when teaching, not when doing actual science. For this class, we are going to use a totally **FAKE** dataset.

[[image:cell_data.png width="291" height="308" align="left"]]In our fake experiment, we are going to test how a drug treatment affects the lifespan of four different types of cells. We measure the following quantities:
# The lifetime of the cell in days ("lifetime").
# The average intracellular calcium concentration in nano-Molars ("[Calcium]").
# The type of cell ("type").
# The diameter of the cell ("diameter") in microns.
# Whether or not that cell was treated ("treated").

The figure to the left is a [[@http://en.wikipedia.org/wiki/Graphical_model|graphical model]]. It shows the dependencies between the variables we've measured. Variables point to other variables that **depend** on them in some way.

From the graphical model, we can determine that the intracellular calcium depends on the diameter of the cell and whether that cell was treated. The lifetime of the cell depends only on calcium concentration and type.


In real life we typically do not have a graphical model for our data, we have to use intuition to guess at one, or use algorithms to construct the most likely one. To learn more about them and also [[@http://en.wikipedia.org/wiki/Bayesian_network|Bayesian Belief Networks]] you can check out [[@http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/090310.pdf|Bayesian Reasoning and Machine Learning (Barber)]].

Here is the code we use to create the fake data:

[[code format="python"]]
import numpy as np
from scipy.stats import *
import pandas as pd
import matplotlib.pyplot as plt

def generate_cell_data(num_samples=100):
    """
        Create fake data that mimics a drug trial on various cell types.
    """

    #probabiity that a cell was treated
    p_treated = 0.5

    #probability mass function for cell type
    type_pmf = np.array([0.20, 0.40, 0.15, 0.25])
    #cdf for cell type
    type_cdf = np.cumsum(type_pmf)

    #mean and std of cell diameter in microns
    diameter_mean = 40.0
    diameter_std = 5.0

    #mean calcium concentration as a function of cell diameter and treatment
    def ca_conc(dia, treatment):
        base_ca = (np.random.randn() + dia) / 6.0
        return base_ca + treatment*np.random.poisson()

    #mean lifetime of cell in days as a function of cell type
    mean_lifetime_by_type = np.array([1.0, 1.5, 2.0, 4.0])

    #create a dictionary of data columns to be turned into a DataFrame
    data = {'treated':list(), 'type':list(), 'diameter':list(), 'calcium':list(), 'lifetime':list()}
    for n in range(num_samples):
        #sample whether or not the cell was treated
        treated = np.random.rand() < p_treated

        #sample the cell type
        cell_type = np.where(type_cdf >= np.random.rand())[0].min()

        #sample the diameter
        dia = np.random.randn()*diameter_std + diameter_mean

        #sample the calcium concentration
        ca = ca_conc(dia, treated)

        #sample cell lifetime
        w1 = 1.0
        w2 = 0.16
        lifetime = np.random.exponential(w1*mean_lifetime_by_type[cell_type] + w2*ca)

        #append the data
        data['treated'].append(treated)
        data['type'].append(cell_type)
        data['diameter'].append(dia)
        data['calcium'].append(ca)
        data['lifetime'].append(lifetime)

    return pd.DataFrame(data)
[[code]]


===1. The Joint and Conditional Probability between Two Random Variables=== 

As we talked about in the last class, the probability of a random variable taking on a value is given by it's probability mass or density function (depending on whether the variable is discrete or continuous). When we have two random variables, we can look at the probability of two values **co-occuring**.

For example, imagine we have data that contains the weather ("rainy", "sunny", "cloudy") and whether flights are delayed ("yes", "no") at an airport. We'll call the random variable for the weather W, and the random variable for whether flights are delayed D. The probability that it is both raining and that there are delays is written like this:

[[math]]
P(W ~=~ \text{rainy}, ~ D ~=~ \text{yes})
[[math]]

To plot joint distributions, we can use a [[@http://matplotlib.org/examples/pylab_examples/hist2d_log_demo.html|2D Histogram]]. (documentation [[@http://matplotlib.org/1.3.1/api/pyplot_api.html#matplotlib.pyplot.hist2d|here]]). In comparison with 2D, a 1D histograms is comprised of a set of **bins**, and each bin represents a range of values in the continuous case, or a single value in the discrete case. Each bin has a **count** of numbers that fall in it's range, and that count is displayed on the y-axis. A 2D histogram has a bin for each **pair** of values, and each bin contains the count of the **number of pairs** that fall within that bin's range. Let's use a 2D histogram to examine the joint probability between diameter and calcium concentration in the fake data:

[[code format="python"]]
def hist2d_plots(df, var1="diameter", var2="calcium"):

    plt.figure()
    plt.hist2d(df[var1], df[var2], bins=[20, 20], normed=True)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.colorbar(label="Joint Probability")

df = generate_cell_data(num_samples=1000)
hist2d_plots(df, var1="diameter", var2="calcium")
plt.show()
[[code]]

Another way to explore the relationship between two variables is to plot one on the x-axis and the other on the y-axis:

[[code format="python"]]
plt.plot(df['diameter'], df['calcium'], 'go')
plt.show()
[[code]]

**EXERCISE**: Explore the relationship between other pairs of continuous random variables, such as calcium and lifetime, diameter and lifetime.

We could use 2D histograms for discrete-continuous or discrete-discrete pairs of random variables, but there are better ways to visualize these types of joint distributions. For discrete-continuous pairs, we can visualize the joint distribution using [[@http://en.wikipedia.org/wiki/Box_plot|Box Plots]]. If we're already working with a Pandas DataFrame, the code is simple:

[[code format="python"]]
df.boxplot('lifetime', by='treated')
[[code]]

We can also overlay two different colored 1D histograms using NumPy:

[[code format="python"]]
def hist_by_discrete(df):
    #select the indices for treated cells
    treated_indices = df['treated'] == True
    plt.figure()
    #create a histogram for untreated cells
    plt.hist(df['lifetime'][treated_indices].values, bins=30, color='b')
    #overlay a histogram for treated cells
    plt.hist(df['lifetime'][~treated_indices].values, bins=30, color='r', alpha=0.75)
    plt.legend(['Treated', 'Untreated'])
    plt.title('Lifetime')

hist_by_discrete(df)
[[code]]

**EXERCISE**: Examine the relationship between cell type and other continuous variables such as calcium and diameter. Is there a relationship?

The joint probability between discrete-discrete pairs of variables can be examined via text, if the number of categories for each variable is small. We can use the Pandas [[@http://pandas.pydata.org/pandas-docs/stable/groupby.html|groupby]] function to look at the counts for each pair:

[[code format="python"]]
g = df.groupby(['treated', 'type'])
print g.size()
[[code]]

We can also get some summary information about the other variables while grouping by pairs of discrete variables:
[[code format="python"]]
print g.agg([len, np.mean, np.std])
[[code]]

If you want to learn more about Pandas, check out Rachel Albert's awesome tutorials [[@python-biophysics-2014/Class 7|here]] and [[@python-biophysics-2014/Class 8|here]].


===2. Independence, Covariance and Correlation=== 

In the last exercise you were asked to speculate as to whether certain variables had relationships. We know from the graphical model that some variables are related, such as diameter and calcium concentration, while others are not, such as type and treatment. When two random variables are not related, they are said to be **independent**. The literal definition of independence is that the value of the joint distribution is equal to the product of the **marginal** distributions:

[[math]]
P(X ~=~ x, ~ Y ~=~ y) = P(X=x)~P(Y=y)
[[math]]

This definition isn't widely used to check for independence. In practice quantities like the **covariance** and the closely related **correlation coefficient** are used.

The [[@http://en.wikipedia.org/wiki/Covariance|covariance]] is a "measure of how two random variables change together". If the covariance has a high absolute value, the two random variables are likely to be dependent on eachother in some way. If the covariance is negative, the two random variables might change in the opposite way, i.e. when one goes up, the other goes down. A covariance of zero indicates that the variables seemingly do not change together. However! [[@http://stats.stackexchange.com/questions/12842/covariance-and-independence|Zero covariance does not imply independence]] even for [[@http://en.wikipedia.org/wiki/Normally_distributed_and_uncorrelated_does_not_imply_independent|Gaussian random variables]].

The [[@http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient|correlation coefficient]] is just the covariance normalized by the standard deviation of each random variable. It ranges from 1 to -1, with 0 meaning that there is no correlation between the two variables. It can be computed using the [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html|corrcoef]] function in NumPy:

[[code format="python"]]
C = np.corrcoef(df['calcium'], df['lifetime'])
cc = C[0, 1]
print cc
[[code]]

Note that corrcoef returns a matrix of values, and we select the upper right entry as the correlation coefficient. It returns a matrix because it supports computing an entire [[@http://en.wikipedia.org/wiki/Covariance_matrix|correlation matrix]]. We'll come back to this when we start doing Linear Algebra.

**EXERCISE**: Compute the correlation coefficient between different pairs of variables. Which variables are correlated? Do the relationships correspond to the connections in the graphical model?


===3. Conditional Probabilities=== 

A [[@http://en.wikipedia.org/wiki/Conditional_probability|conditional probability]] is when we look at the distribution of a single random variable **given** the value of another. A simple example:

[[code format="python"]]
df.hist('lifetime', by='type')
[[code]]

That plot show 4 histograms - one for each value of cell type. Each conditional probability would be written and read like this:

[[math]]
P( \text{lifetime} ~|~\text{type}=0 ) ~=~ \text{"probability of lifetime given type equals zero"}
[[math]]

Conditional probabilities can be computed from joint probabilities:

[[math]]
P( \text{lifetime} ~|~ \text{type}=0 ) ~=~ \frac{P( \text{lifetime}, ~\text{type}=0 )}{P( \text{type}=0 )}
[[math]]

We'll use conditional probabilities below when we talk about maximum likelihood. You should spend some time on your own reading about [[@http://en.wikipedia.org/wiki/Bayes%27_theorem|Bayes Theorem]], a very important theorem! We'll also use conditional probabilities when we talk about generalized linear models.


===4. Fitting Data, Parameters, and Likelihood=== 

Say we would like to fit the lifetime column data to an [[@http://en.wikipedia.org/wiki/Exponential_distribution|exponential distribution]]. The exponential distribution can be used to model the time between successive events. It has has a parameter named **rate**, which is equal to the average number of events that happen in a unit of time. To **fit** an exponential distribution with the data means to algorithmically find the value for the rate that does the **best job** at modeling the data.

In order to algorithmically determine what is "best", we need an **objective** **function** that quantifies how well the data is fit given a proposed rate. Statisticians determined that the optimal function to do this is the [[@http://en.wikipedia.org/wiki/Likelihood_function|likelihood function]]. The likelihood function is defined as the probability of seeing the data given the proposed parameter.

What is the probability of seeing the data? In our case, we'll call the random variable that represents cell lifetime X. Each data point, an observed value of the random variable, will be a little x with a subscript. A dataset of N samples, which we'll call D, is written mathematically like this:

[[math]]
\mathcal{D} ~=~ \{x_1, ~ x_2, ~ ..., ~ x_N\}
[[math]]

The probability of an individual data point is equal to the probability density function evaluated at that point given the proposed parameter:

[[math]]
P( X ~=~ x_i ~|~ \lambda ) ~=~ \text{pmf}(x_i) ~=~ \lambda ~ e^{-\lambda x_i}
[[math]]

The [[@http://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables|IID assumption]] is frequently made that states each data point is independent of all the other data points. Remember from earlier - if two random variables are independent from each other, then their joint probability is the product of their individual probabilities. If we make the assumption that each data point is independent from the other data points, then **their joint probability is equal to the product of their individual probabilities**. So we can write down the equation for the likelihood:

[[math]]
\text{likelihood}(\lambda) ~=~ P( \mathcal{D} ~|~ \lambda) ~=~ P( \{x_1, ~ x_2, ~ ..., ~ x_N\} ~|~ \lambda) ~=~ \prod_{i=1}^N P( X ~=~ x_i ) ~=~ \prod_{i=1}^N ~ \lambda ~ e^{-\lambda x_i}
[[math]]

For many commonly used distributions, such as the exponential, the likelihood can be written down. Also, sometimes we can use calculus to determine the value of the parameter that **maximizes the likelihood**. A parameter that maximizes the likelihood is the **best fit for the data**.

In this example we're going to explore how different values of the rate affect the likelihood of the data. Here's some code to do it:

[[code format="python"]]
def lifetime_likelihood(df, rate):
    """ Computes the likelihood of the lifetime column given a rate. """

    #get an array of data points
    x = df['lifetime'].values
    #compute the likelihood
    likelihood = np.log(rate*np.exp(-rate*x)).sum()
    #compute the maximum likelihood estimate of the rate
    best_rate = 1.0 / x.mean()
    max_likelihood = np.log(best_rate*np.exp(-best_rate*x)).sum()
    print 'Log Likehood for rate={:.4f}: {}'.format(rate, likelihood)
    print 'Max Log Likehood for rate={:.4f}: {}'.format(best_rate, max_likelihood)

    plt.figure()
    #plot the data distribution
    plt.hist(x, bins=30, color='k', normed=True)
    plt.xlabel('Lifetime')

    #plot the pdf of an exponential distribution fit with max likelihood
    rv = expon(best_rate)
    xrng = np.linspace(0, x.max(), 200)
    plt.plot(xrng, rv.pdf(xrng), 'r-', linewidth=2.0)

    #plot the pdf of an exponential distribution with given rate
    rv = expon(rate)
    plt.plot(xrng, rv.pdf(xrng), 'b-', linewidth=2.0)
    plt.legend(['Max likelihood', '$\lambda$={:.4}'.format(rate), 'Data'])

lifetime_likelihood(df, 5.0)
[[code]]

**EXERCISE**: Explore different values for the rate, some that are close to the maximum likelihood solution, some that are far away. Can you find a parameter that produces a higher value for the log likelihood than the maximum likelihood parameter produces?
