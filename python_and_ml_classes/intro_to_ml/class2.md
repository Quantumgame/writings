===Outline=== 

1. Probability Theory in a Nutshell
2. The Uniform Distribution
3. The Bernoulli (Binary) Distribution
4. The Gaussian Distribution
5. The Poisson Distribution


<span style="color: inherit; font-family: inherit; font-size: 1.1em; line-height: 1.1;">**1. Probability Theory in a Nutshell**</span>

There is a great Khan Academy series that can help with this topic: [[@https://www.khanacademy.org/math/probability/random-variables-topic/random_variables_prob_dist/v/random-variables|Random Variables and Probability Distributions]].

A **random variable** is an abstract representation of an event that can take on random values. Some examples:

* The number of white blood cells in the human body.
* The voltage measured from an electrode over the course of an experiment.
* The DNA sequence of a gene.
* The value of a pixel in a set of images.
* The amplitude of air pressure on an eardrum in a room full of conversations.
* The number of Tuesdays in a year that a giraffe at the zoo will be videotaped [[@https://www.youtube.com/watch?v=vEkIBK40AN4|licking it's lips]].

Random variables can be one of two types:

# **Discrete**: the random variable takes on numbers that can be counted (even if there are an infinite number of them).
# **Continuous**: the random variable can take on any decimal number in a range.

<span style="line-height: 1.5;">Probability theory gives us a way of quantifying randomness. In the previous class, we used a **Gaussian random variable** to simulate "noise" and added it to line and curves. What dictates the range of those numbers that a random variable can take on? Do some numbers appear more than others?</span>

The answer to these questions is that the range and frequency at which certain random numbers appear depend on the **probability distribution** of the random variable. Probability distributions look like this:
[[image:all_dists.png width="668" height="191" align="center"]]
<span style="line-height: 1.5;">The </span>**<span style="line-height: 1.5;">x-axis</span>**<span style="line-height: 1.5;"> is the </span>**<span style="line-height: 1.5;">value</span>**<span style="line-height: 1.5;"> that the random variable can take on, the </span>**<span style="line-height: 1.5;">y-axis</span>**<span style="line-height: 1.5;"> is the </span>**<span style="line-height: 1.5;">probability</span>**<span style="line-height: 1.5;"> that the random variable takes on that value.</span>

There are several numbers we'll use to summarize a distribution:
# The **mean** is the average value that the random variable takes.
# The **variance** is how much the random variable spreads out, giving an indication of it's range.
# The **median** is the value on the x-axis that splits the distribution in half.

To generate a random number is to **sample** a random variable from a probability distribution. We'll go over several important random variables. For each random variable, there are five things to we will learn how to do :
# **Compute and visualize** the probability density by building a **histogram** of the random variable to distribution and plotting it.
# **Compute the probability** of **events** regarding that random variable.
# **Sample** from the distribution to generate random numbers.
# **Parameterize** the distribution to specify it's shape.
# Compute the **mean** and **variance** of the random variable.

These things will be very easy to do because we'll be using the [[@https://docs.scipy.org/doc/scipy-0.9.0/reference/stats.html|scipy.stats]] module.


===2. The Uniform Distribution=== 

The distribution of a **uniform random variable** is flat - any value between 0 and 1 is equally likely to occur. There are an infinite number of possibilities between 0 and 1 - uniform random variables are **continuous**. We would like to sample from the uniform distribution and plot it. In order to do this, we need to generate a large number of random numbers from the uniform distribution and keep track of how many times each number occurs. We will accomplish this by creating a [[@http://en.wikipedia.org/wiki/Histogram|histogram]] of the numbers we sample.

Let's get to some coding - in this example we generate a bunch of random numbers from a uniform distribution and plot it's histogram:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

#the scipy.stats package contains many different
#types of random variables that we can choose from
from scipy.stats import *

def plot_uniform(num_samples=10000):
    """ Create a uniform random variable and make a histogram.

        num_samples: The number of random numbers to generate.
    """

    #create a uniform random variable
    rv = uniform()

    #generate a bunch of random numbers from rv
    sample_data = rv.rvs(num_samples)

    #create a histogram
    plt.figure()
    plt.hist(sample_data, bins=30, color='r')
    plt.title('The Uniform Distribution (N={0})'.format(num_samples))
    plt.xlabel('Value')
    plt.ylabel('Count')

    plt.show()

#plot a uniform distribution with 1000 samples
plot_uniform(1000)
[[code]]

**EXERCISE**: How many samples do we need to make the histogram look flat? What does the histogram look like when there are only 100 samples?

It's now very important to introduce two types of functions that involve a probability distribution. The first is the **[[@http://en.wikipedia.org/wiki/Probability_density_function|probability density function]]** (pdf). The pdf measures the probability that a random variable takes on a given value. Let the symbol "X" represent a uniform random variable, and imagine that we want to compute the probability that X is equal to 0.75. We could write that as:

[[math]]
\text{pdf}(0.75)~ = ~P(X = 0.75) ~=~ "probability~that~X~is~0.75"
[[math]]

The **[[@http://en.wikipedia.org/wiki/Cumulative_distribution_function|cumulative distribution function]]** (cdf) tells us the probability that X is **less than or equal to** a given number. We we'll write it as:

[[math]]
\text{cdf}(0.75) ~=~ P(X \leq 0.75) ~=~ "probability~that~X~is~less~than~or~equal~to~0.75"
[[math]]

The funny thing about pdfs, is that the probability of X being *exactly* 0.75 is very very very small. We really want to compute the probability that X is *close* to 0.75, say within 0.01. That means that we want the probability that X is greater than or equal to 0.74, and less than or equal to 0.76. A fancy way of writing this is:

[[math]]
P(~|X - 0.75| \lt 0.01~) ~=~ P(~0.74 \lt X \leq 0.76~) ~=~"probability~that~X~is~within~0.01~of~0.75"
[[math]]

If you're into calculus, it turns out that **the derivative of the cdf is the pdf**. Without explaining here, this implies that we can compute the aforementioned probability very easily, once we have the cdf:

[[math]]
P(~|X - 0.75| \lt 0.01) ~=~ \text{cdf}(0.76) - \text{cdf}(0.74)
[[math]]

A random variable can be defined by it's pdf or cdf. The cdf is the more proper and general way of defining a probability distribution that always works. If you know a random variable's cdf, then you know everything you need to about that random variable.

If I asked you to tell me the probability that a uniform random number was between 0.25 and 0.40, you could now use the scipy.stats package to compute that for me. Here's code that does it:

[[code format="python"]]
def compute_probability_of_range(start, end):
    """ Compute the probability of a uniform random variable
        falling between start and end.
    """
    #create a uniform random variable
    rv = uniform()

    #compute the difference in cdf between the two points
    p = rv.cdf(end) - rv.cdf(start)

    return p

#compute the probability that a uniform random variable is between 0.25 and 0.40
compute_probability_of_range(0.25, 0.40)
[[code]]

**EXERCISE**: Compute the probability for various ranges of values - do you see a pattern?

Without much explanation for this example - the mean of the uniform distribution is 1/2, the variance is 1/12.

FUN FACT: As long as you can generate random numbers from the uniform distribution, you can generate random numbers from any arbitrary distribution so long as you have it's cdf, using the [[@http://en.wikipedia.org/wiki/Inverse_transform_sampling|Inverse Transform]] method.


===3. The Bernoulli (binary) Distribution=== 

The uniform distribution provides the simplest version of a continuous random variable, and the [[@http://en.wikipedia.org/wiki/Bernoulli_distribution|Bernoulli distribution]] is the simplest version of a **discrete** random variable. It takes on a value of either 0 or 1. The probability of 1 is written as a (non-random) variable named p. The probability measure has to sum to 1, so the probability of a 0 is 1-p.

Discrete variables don't have probability density functions, they have probability **mass** functions (pmfs). The pmf can tell us the probability that a random variable is exactly equal to something. If we re-use X from the uniform distribution example to represent a random binary variable, we can write it as:

[[math]]
P(X = 1) ~=~ \text{pmf}(1) ~=~ p
[[math]]

[[math]]
P(X = 0) ~=~ \text{pmf}(0) ~=~ 1-p
[[math]]

We can't define a binary random variable without specifying p! p is considered a **parameter** of the distribution. Let's go ahead and generate a binary random variable and sample from it:

[[code format="python"]]
import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt

def plot_bernoulli(p, num_samples=1000):
    """
        Plot a histogram of a Bernoulli distribution.

        p: The probability of emitting a 1.
    """

    #p is the probability of emitting a one
    rv = bernoulli(p)

    #generate random bernoulli numbers and compute the mean
    sample_data = rv.rvs(num_samples)
    sample_mean = sample_data.mean()
    print 'Sample mean: {:.2f}'.format(sample_mean)
    print 'True mean: {:0.2f}'.format(p)

    #make a histogram of the sample data
    plt.figure()
    plt.hist(sample_data, color='g')
    plt.xlim(-0.5, 1.5)
    plt.title('The Bernoulli Distribution (p={:.2f}, N={})'.format(p, num_samples))
    plt.xticks([0, 1], [0, 1])
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.show()

plot_bernoulli(0.25, num_samples=1000)
[[code]]

**EXERCISE**: Convince yourself that the world makes sense by playing around with values of p and the number of samples.


===4. The Gaussian Distribution=== 

[[@http://en.wikipedia.org/wiki/Carl_Friedrich_Gauss|Carl Gauss]] straddled the 18th and 19th centuries and was one of the greatest mathematicians of all time. People love that guy. There's a bunch of stuff named after him, including the absolutely prolific [[@http://en.wikipedia.org/wiki/Normal_distribution|Gaussian Distribution]]. There are many reasons that the Gaussian (or Normal) distribution is so widely used, here are a few:
# The sum of many random variables tends to be Gaussian distributed.
# The Gaussian distribution is the simplest distribution to assume when all we have is a mean and variance measured from our sample.
# It can be completely described by only it's mean and variance.

The distribution is continuous, so it has a probability density function. There is a very clear equation for it:

[[math]]
P(X = r) ~=~ pdf(x) ~=~ \frac{1}{\sigma \sqrt{2\pi}} exp \left( - \frac{1}{2} \frac{(r - \mu)^2}{2 \sigma^2} \right)
[[math]]

There are two parameters of the distribution:

[[math]]
\mu ~=~ \text{the mean}
[[math]]
[[math]]
\sigma ~=~ \text{the standard deviation}
[[math]]

Note that the standard deviation is the square root of the variance.

Let's make a plot of the probability density function:

[[code format="python"]]
import numpy as np
from scipy.stats import *
import matplotlib.pyplot as plt

def plot_gaussian(mean, std, num_samples=10000):

   #create a gaussian random variable
   rv = norm(loc=mean, scale=std)

   #generate random Gaussian numbers and compute the mean
   sample_data = rv.rvs(num_samples)
   sample_mean = sample_data.mean()
   print 'Sample mean: {:.2f}'.format(sample_mean)
   print 'True mean: {:0.2f}'.format(mean)

   sample_data = rv.rvs(num_samples)
   sample_std = sample_data.std()
   print 'Sample std: {:.2f}'.format(sample_std)
   print 'True std: {:0.2f}'.format(std)

   #make a histogram of the sample data
   plt.figure()
   plt.hist(sample_data, color='c', bins=30)
   plt.title('The Gaussian Distribution ($\mu$={:.2f}, $\sigma$={:.2f}, N={})'.format(mean, std, num_samples))
   plt.xlabel('Value')
   plt.ylabel('Count')
   plt.show()

plot_gaussian(0.0, 1.0, num_samples=10000)
[[code]]

**EXERCISE**: Play with the mean, standard deviation, and sample size until it looks like data you've collected in your lab.

**EXERCISE**: Imagine that the current that flows under the Bay Bridge is Gaussian distributed, with a mean of zero, measured in meters/second. Negative current flows out of the bay, positive current flows into the bay. It turns out that if the current exceeds 3 m/s, it's so strong that it pulls great white sharks into the bay, and they will eat people with small sailboats. What is the probability that the current will exceed 3m/s?

Exercise Hint: Use the cdf and the following relationship to solve the problem:
[[math]]
P(X \leq r) ~=~ 1 - P(X \gt r)
[[math]]


===5. The Poisson Distribution=== 

Did you know that "poisson" means "fish" in French? This distribution was named after [[@http://en.wikipedia.org/wiki/Sim%C3%A9on_Denis_Poisson|Mr. Fish]], an awesome mathematician. It's a discrete distribution that models variables that represent counts. So it's always positive. Whether you're measuring the number of photons that bounce off your crazy solid state physics thing, or the number of insects per night that a bat eats, you can use this distribution. The pmf is given as:

[[math]]
P(X = k) ~=~ \frac{\lambda^k e^{-k}}{k!}
[[math]]

There is one parameter, lambda, the "rate" of the distribution:
[[math]]
\lambda ~=~ \text{the rate}
[[math]]

The higher the lambda, the higher the probability of large counts. Let's simulate it and build some intuition:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

def plot_poisson(rate, num_samples=10000):

   #generate random Poisson numbers and compute the mean
   sample_data = np.random.poisson(rate, size=num_samples)
   sample_mean = sample_data.mean()
   print 'Sample mean: {:.2f}'.format(sample_mean)
   print 'Sample variance: {:.2f}'.format(sample_data.var())
   print 'True rate: {:0.2f}'.format(rate)

   sample_max = sample_data.max()

   #make a histogram of the sample data
   plt.figure()
   plt.hist(sample_data, color='c', bins=30)
   plt.title('The Poisson Distribution ($\lambda$={:.2f}, N={})'.format(rate, num_samples))
   plt.xlabel('Value')
   plt.ylabel('Count')
   plt.xticks(np.arange(sample_max+1), np.arange(sample_max+1))
   plt.axis('tight')
   plt.show()

plot_poisson(20.0, num_samples=10000)
[[code]]

**EXERCISE**: Play around with lambda and see what you get. Do you notice the special relationship between the mean and the variance?
