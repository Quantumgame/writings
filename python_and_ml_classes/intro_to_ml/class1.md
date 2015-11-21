//Code for this class can be found [[@https://github.com/mschachter/ml-biophysics-2014/tree/master/class1|here]].//
=== === 

===Welcome to Machine Learning!=== 

<span style="line-height: 1.5;">The first thing you need for this class is a good Python installation. I would recommend either </span><span style="background-color: #ffffff; line-height: 1.5;">[[@http://continuum.io/downloads|Anaconda]]</span><span style="line-height: 1.5;"> or </span><span style="background-color: #ffffff; line-height: 1.5;">[[@https://store.enthought.com/|Canopy Express]]</span><span style="line-height: 1.5;"> . Installing either of these packages will give you access to the following essential libraries:</span>
* [[@http://numpy.org|NumPy]]
* [[@http://scipy.org|SciPy]]
* [[@http://matplotlib.org|Matplotlib]]
* [[@http://pandas.org|Pandas]]
* [[@http://scikit-learn.org|scikit-learn]]

<span style="line-height: 1.5;">The next thing you need is to know some Python. If you're new to Python, you can find some tutorials and resources in the 5-week </span><span style="background-color: #ffffff; line-height: 1.5;">[[@python-biophysics-2014/home|Intro to Python]]</span><span style="line-height: 1.5;"> that preceded this class. Also become comfortable with [[@python-biophysics-2014/Class 2|using the terminal]], [[@python-biophysics-2014/Class 5|NumPy]], and read a bit about [[@http://scikit-learn.org/stable/tutorial/basic/tutorial.html|ML using scikit-learn]].</span>

After that, you should set yourself up with a few good Machine Learning books, online classes, and tutorials. Check out the [[Big List of Machine Learning Resources and Software|Big List of Machine Learning Resources]] for some of that.

Although Mathematics is vast and ever-expanding, there's a reasonably tight core of math that we'll be relying on to do most of the work and explanations. The biggest components of Machine Learning are Linear Algebra and Probability Theory, which we will be spending plenty of time on.


===What is Data? What are Features?=== 

Data is diverse, but basically it's anything that can be quantified. The data types that we'll be dealing with in this class can be broken down into two groups:
# **Real-valued (continuous)**: data that takes the form of a floating point number.
# **Categorical (discrete)**: unordered data that takes the form of a discrete category. Examples include binary data, which takes on the value of 0 or 1, and multi-class data, which can take on any value of a finite set of categories, such as "cat", "dog", "person" or "A", "B", "C".

Some real world examples of data:
* The text that comprises an email, used to create a spam filter (categorical).
* The pixels of an image for a handwritten digit, used for digit recognition (real-valued).
* Recordings of human speech, used for speech recognition (real-valued).
* The sequence of nucleotide letters that comprise a gene, used to predict treatment outcomes for experimental drugs (categorical).
* An image of a [[@http://en.wikipedia.org/wiki/DNA_microarray|microarray]], used to predict the expression levels of different genes.

The examples above are **raw data**, which is data in it's most primitive form. Raw data is typically **preprocessed** into **features**, and algorithms are built that utilize those features. Some examples of how data could be preprocessed into features:
* Quantify each word of the text of an email into a [[@http://www.tfidf.com/|TF-IDF]] number, and form a list of those numbers, one for each word.
* Convert each image of a handwritten digit into a set of [[@http://en.wikipedia.org/wiki/Scale-invariant_feature_transform|SIFT]] features.
* Examine the time-varying frequencies of human speech by transforming it into [[@http://en.wikipedia.org/wiki/Mel-frequency_cepstrum|MFCC]] features.

So - **data** is anything that can be quantified, and we **preprocess** the data to produce **features** which will simplify analysis.


===What is Machine Learning?=== 

Machine Learning is a field where people develop algorithms that automatically learn patterns from data. Here are a bunch of examples:
* From an image of a handwritten digit, predict what that digit is.
* From human speech, decode what words were spoken.
* Given the measurement of various plant properties, cluster the plants into different types without previous knowledge of the types.
* Given a set of images, identify a region of each image that contain a human face.
* Given brain activity recorded from a region of the brain, predict what a person is thinking.

Machine Learning can be broken down into a few different groups of algorithms, which at times can overlap:
# **Supervised Learning**: you know what your input data is, you break it down into features, and you want to predict some sort of output from your input features. If the output is real-valued, the type of **prediction** you are doing is called **regression**. If the output is categorical, you are doing **classification**. In supervised learning, your output data is clearly **labeled**, so that an **error function** can be computed that tells the algorithm how well it does at prediction.
# **Unsupervised Learning**: you have a bunch of input data, and you either want to **cluster** it into meaningful groups, or perhaps you want to do **feature learning** without any prior knowledge of the type of features you're looking for.
# **Reinforcement Learning**: you have an **agent**, which is a piece of software or a robot that performs **actions**, and you want to find the actions that maximize some sort of **reward** function.

We are only going to talk about Supervised and Unsupervised learning in this class, and mostly Supervised.


===Our First Machine Learning Problem: Fitting a Noisy Line=== 

Almost everything we need to know about this problem comes from a long-forgotten high school algebra class. Do you remember what the equation for a straight line is? It's this:

[[math]]
f(x) = mx + b
[[math]]

Where m is the **slope** and b is the **intercept**. Let's write some code that plots a few different lines:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

#create an array of points to evaluate the line
num_samples = 100
x = np.linspace(-5, 5, num_samples)

#set the value of the slope and intercept
slopes = [-1.0, 0.0, 1.0, 2.0]
intercepts = [0.25, -0.5, 1.25, 0.6]

#open up a blank figure
plt.figure()
#plot a line for each slope and intercept value
for m,b in zip(slopes, intercepts):
    #compute the value of the line at each point
    y = m*x + b
    #make a plot of the line
    plt.plot(x, y, linewidth=2.0)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axis('tight')
plt.show()
[[code]]

**EXERCISE**: play around with different values of slopes and intercepts to gain intuition about how the line depends on them.

Real data is **noisy**. This means that the strict relationship between input and output is corrupted by random numbers. We represent this mathematically as:

[[math]]
f(x) = mx + b + noise
[[math]]

where "noise" is a **random variable**, which can take on a random value for every data point. Very often the assumption is made that this random number comes from a [[@http://en.wikipedia.org/wiki/Normal_distribution|Gaussian Distribution]], which we will spend alot more time on soon.

In real life, data is noisy, and we often don't have many **samples**. Let's simulate this situation for a single slope and intercept:

[[code format="python"]]
#create an array of points to evaluate the line
num_samples = 10
x = np.linspace(-5, 5, num_samples)

#get the number of samples and print it
num_samples = len(x)
print("# of samples: {0}".format(num_samples))

#set the value of the slope and intercept
m = 2.0
b = -0.9

#create an array of Gaussian random noise, one
#random number per sample
noise = np.random.randn(num_samples)

#create a noiseless line
y = m*x + b
#create a noisy line
ynoisy = m*x + b + noise

#open up a blank figure
plt.figure()
#plot the non-noisy line
plt.plot(x, y, 'k-', linewidth=2.0)
#plot the noisy line
plt.plot(x, ynoisy, 'go')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axis('tight')
plt.show()
[[code]]

In the example above, we knew the true values of the **parameters** m and b. But what if we didn't? We would need to take the data and **fit** it using an algorithm. Luckily this is easy to do programmatically, we just need to use NumPy's [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html|polyfit]] function. "poly" stands for polynomial. A straight line is a **first degree polynomial**, it doesn't have any terms that involve x^2, x^3, etc. This code is a continuation of the last bit of code:

[[code format="python"]]
#now fit the noisy line using polyfit
fitted_slope, fitted_intercept = np.polyfit(x, ynoisy, deg=1)

print("Fitted slope={0:.2f}, Predicted slope={1:.2f}".format(fitted_slope, m))
print("Fitted intercept={0:.2f}, Predicted intercept={1:.2f}".format(fitted_intercept, b))

#compute the line predicted by polyfit
ypredicted = fitted_slope*x + fitted_intercept

#plot the actual line, the predicted line, and the data points
plt.figure()
#plot the non-noisy line
plt.plot(x, y, 'k-', linewidth=2.0)
#plot the predicted line
plt.plot(x, ypredicted, 'r-')
#plot the noisy data
plt.plot(x, ynoisy, 'go')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axis('tight')
plt.show()
[[code]]

**EXERCISE**: Note that polyfit does not figure out the true values of the slope and intercept, it **approximates** them. In Machine Learning, **the more data you have, the better the predictor**. Inspect the accuracy of the fitted values of slope and intercept as you increase the number of samples (by decreasing the spacing between points in x, the third argument in the expression np.arange(-5, 6, 1)).


===Our Second Machine Learning Problem: Fitting a Noisy Polynomial=== 

Do you remember [[@http://en.wikipedia.org/wiki/FOIL_method|FOIL]]? It's a way to multiply two simple algebraic expressions, and goes like this:
[[math]]
(x - 4) (x + 3)
[[math]]
[[math]]
= x^2 + 3x - 4x -12
[[math]]
[[math]]
= x^2 - x -12
[[math]]

The last expression is a **second-degree polynomial**! A **third-degree polynomial** would have a x-cubed term in it, and so forth. Did you know that you can approximate any function using a polynomial of high enough degree? There's a [[@http://mathworld.wolfram.com/WeierstrassApproximationTheorem.html|theorem]] that says so. Of course, that polynomial might be of extremely high (even infinite) degree! People use [[@http://www.math.ucla.edu/~baker/149.1.02w/handouts/dd_splines.pdf|Cubic Splines]] to get around that problem.

**EXERCISE**: Experiment with random polynomials. Copy and paste the function below into a script or IPython, and call the function with various values for the degree and noise standard deviation, plotting the results.

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

def evaluate_polynomial(x, coefficients):
    """ Evaluate a polynomial.

        x: a numpy.array of points to evaluate the polynomial at.

        coefficients: a numpy.array of coefficients. Don't forget
            the coefficient for the bias term! The degree of the
            polynomial is len(coefficients)-1.

        Returns: y, a numpy.array of values for the polynomial.
    """

    #initialize the output points to zero
    num_samples = len(x)
    y = np.zeros([num_samples])

    deg = len(coefficients)-1
    for k in range(deg+1):
        y += coefficients[deg-k] * x**k

    return y

def generate_polynomial(deg=3, noise_std=1e-1, num_samples=100):
    """ Generate the outputs of a noisy polynomial.

        x: A numpy.array of points on the x-axis to
            evaluate the polynomial at.

        deg: The degree of the polynomial, defaults to 1.

        noise_std: The standard deviation of the Gaussian noise
            that is added to each sample. If zero, no noise will
            be added. The higher the standard deviation, the more
            the polynomial will be drown out in noise.

        returns: coefficients,y - An array of coefficients (from
            the highest degree to lowest), and an array of points
            where the polynomial was evaluated.
    """

    #generate the points to evaluate the polynomial
    x = np.linspace(-1, 1, num_samples)

    #generate random coefficients from a Gaussian
    #distribution that has zero mean and a standard
    #deviation of 1.
    coefficients = np.random.randn(deg+1)

    y = evaluate_polynomial(x, coefficients)

    #create and add the noise
    noise = np.random.randn(len(x))*noise_std
    y += noise

    return x,y,coefficients

#generate a 3rd degree polynomial and plot it
x,y,coef = generate_polynomial(deg=3, noise_std=1e-1, num_samples=100)
plt.figure()
plt.plot(x, y, 'k-')
plt.axis('tight')
plt.show()
[[code]]


**EXERCISE**: Use the additional code below to generate and fit a 3rd degree polynomial. Examine how the fit changes when you change the number of samples, noise standard deviation, and degree of the polynomial used to fit. You should try to observe the following:
# The fit is worse when the data is noisy or when the number of samples is decreased.
# As the degree of the polynomial used to fit the data is increased, it does good at fitting the actual data points, but much worse at fitting the actual function! This is called **overfitting**, the high degree polynomial does not **generalize** to predict points on the curve it has not observed from the data.

[[code format="python"]]
def fit_and_plot(degree_of_actual=3, degree_of_fit=3,
                 num_samples=10, noise_std=1e-1):
    """ Generates a random noisy polynomial and fits it using polyfit.

        degree_of_actual: The degree of the generated polynomial.

        degree_of_fit: The degree passed to polyfit used to fit the
            generated polynomial.

        num_samples: The number of samples generated for the polynomial.

        noise_std: The standard deviation of the noise added to the
            generated polynomial.
    """

    #generate a polynomial
    x,y,coefficients = generate_polynomial(deg=degree_of_actual,
                                           noise_std=noise_std,
                                           num_samples=num_samples)

    #fit the polynomial
    fit_coefficients = np.polyfit(x, y, deg=degree_of_fit)

    #evaluate the fit polynomial
    fit_y = evaluate_polynomial(x, fit_coefficients)

    #create a dense set of points on which both polynomials
    #will be plotted and evaluated
    dense_x = np.linspace(-1, 1, 100)

    plt.figure()
    #plot the generated polynomial
    plt.plot(dense_x, evaluate_polynomial(dense_x, coefficients), 'k-')
    #plot the fit polynomial
    plt.plot(dense_x, evaluate_polynomial(dense_x, fit_coefficients), 'r-')
    #plot the data points
    plt.plot(x, y, 'bo')
    plt.axis('tight')
    #make a legend
    plt.legend(['Actual', 'Fit'])
    plt.show()

fit_and_plot(degree_of_actual=3, degree_of_fit=3, num_samples=10, noise_std=1e-1)
[[code]]


===Homework=== 
* Skim chapter 1 of PRML. Start reading chapter 2 if you'd like.
* If you're having trouble with the programming aspects, review some material from the [[@python-biophysics-2014/home|Intro to Python]] course.
* Instead of using polyfit, try using cubic splines with the function [[@http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d|scipy.interpolate.interp1d]] from SciPy's [[@http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/interpolate.html|interpolate]] package. If you're having trouble, check out this [[@http://stackoverflow.com/questions/11851770/spline-interpolation-with-python|Stack Overflow post]].

