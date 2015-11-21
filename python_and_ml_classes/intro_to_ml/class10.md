1. Generating Time Series: Sine Waves, Difference Equations, Autoregressive Models
2. Fourier Transforms and Power Spectrums
3. Auto Correlation and Cross Correlation Functions
4. Linear Filters


===1. Generating Time Series: Sine Waves, Difference Equations=== 

A **time series** is the measurement of a number or vector over a series of time points. Here are some examples:

# The voltage recorded from the membrane of a neuron every microsecond while a 3 second long stimulus is being played to the animal.
# The sound pressure recorded at the ear drum recorded every 22 microseconds for a one hour performance of an extraordinary gathering of hip hop musicians.
# The value of the Dow Jones Industrial Average every day for a year.
# The musical notes hit on a piano every beat during the playing of a song.
# The weather (rainy, sunny, cloudy) recorded once a day for a year.
# The frames of a raw video sequence.

In each of these examples, the time series has a **duration** and also a **sampling rate**, which is the number of samples per second that the series is recorded for. The first three examples are **scalar** time series that take on a single value at each time point. The fourth example, the time-varying key presses on a piano, is a **multivariate** (or vector) time series. Each time point can be represented by an 88 dimensional binary vector, where a one indicates that note was played at that time. The 5th example is a **discrete** time series that takes on a categorical value at each point in time.

====Sine Waves==== 

Time series are represented mathematical simply as **functions of time**. Let's begin with the equation for a [[@http://en.wikipedia.org/wiki/Sine_wave|sine wave]]:

[[math]]
y(t) ~=~ \text{sin} \left( 2 \pi f t \right)
[[math]]

The variable f is the frequency, in units of Hertz (samples per second). Let's mess around with sine waves in NumPy:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

def generate_sine_wave(freqs=[1.0, 2.0, 5.0], duration=5.0, sample_rate=1e3, plot=True):

    #generate vector that represents time
    num_samps = int(duration*sample_rate)
    t = np.arange(num_samps) / sample_rate

    #generate sine wave
    y = np.zeros([len(t)])
    for freq in freqs:
        y += np.sin(2*np.pi*t*freq)

    if plot:
        #plot the sine wave
        plt.figure()
        plt.plot(t, y, 'c-', linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    return t,y

generate_sine_wave(freqs=[1.0, 2.0, 5.0], duration=5.0, sample_rate=1e3)
plt.show()
[[code]]

**EXERCISE**: Play around with different frequencies for the sine wave!


====Difference Equations==== 

Another perspective on time series comes from the field of [[@http://en.wikipedia.org/wiki/Dynamical_system|dynamical systems]]. In Dynamical Systems Theory, a time series is generated from an ongoing **process** that can be fully described at a given point in time by it's **state vector**, an N-dimensional vector which we'll call **x**. When N=1, the transition from the current state at time t to the next state at time t+1 can be written down as a [[@http://www.math.cornell.edu/~lipa/mec/lesson2.html|one dimensional map]], also called a **difference equation**:

[[math]]
x_{t+1} ~=~ f \left( x_t \right)
[[math]]

By running the difference equation for many time steps and recording the state vector at each step, we can generate a time series. This time series is often called the **state** **trajectory** of the system. Here's some code to generate time series from difference equations:

[[code format="python"]]
def run(transition_function, initial_value, nsteps=100):
    """ Simulate a system using a difference equation.

        transition_function: The right hand side of the difference equation.
        initial_value: The starting value for the state.
        nsteps: The number of steps to run the system for.
    """

    x = np.zeros([nsteps])
    x[0] = initial_value
    for k in range(1, nsteps):
        x[k] = transition_function(x[k-1])

    return x


def difference_equation_examples(logistic_map_r=3.86):
    """ Show some examples of difference equations.

        logistic_map_r: The value of r to give the logistic
            map in the third plot. Defaults to 3.86, which
            creates chaotic dynamics.
    """

    plt.figure()

    plt.subplot(3, 1, 1)
    x = run(np.cos, initial_value=1.0, nsteps=20)
    plt.plot(x, 'k-')
    plt.axis('tight')
    plt.title('$x_{t+1} = cos(x_t)$')

    plt.subplot(3, 1, 2)
    r = 3.57
    logistic_map = lambda x: r*x*(1.0 - x)
    x = run(logistic_map, initial_value=0.5, nsteps=75)
    plt.plot(x, 'k-')
    plt.axis('tight')
    plt.title('$x_{t+1} = %0.6f x_t (1 - x_t)$' % r)

    plt.subplot(3, 1, 3)
    r = logistic_map_r
    logistic_map = lambda x: r*x*(1.0 - x)
    x = run(logistic_map, initial_value=0.5, nsteps=175)
    plt.plot(x, 'k-')
    plt.axis('tight')
    plt.title('$x_{t+1} = %0.6f x_t (1 - x_t)$' % r)

difference_equation_examples(logistic_map_r=3.86)
plt.show()
[[code]]

**EXERCISE**: The two bottom plots are trajectories of the [[@http://en.wikipedia.org/wiki/Logistic_map|logistic map]]. The logistic map can exhibit [[@http://en.wikipedia.org/wiki/Chaos_theory|chaotic]] behavior for certain values of it's parameter, r. Play with different values for logistic_map_r to explore the variety of series that can be generated.


====Auto-regressive Models==== 

[[@https://www.otexts.org/fpp/8/3|Autoregressive Models]] are difference equations where the next state is a linear combination of the previous states. The difference equation for them looks like this:

[[math]]
y(t) ~=~ w_1 ~ y(t-1) ~+~ w_2 ~ y(t-2) ~+~ \cdots ~+~ w_n ~ y(t-n)
[[math]]

The w terms are weights multiplied by the history of y. We can start the series off at time 0 with some random initial value and then see what happens with it. Let's do that in NumPy:

[[code format="python"]]
def generate_ar_process(weights=[-0.3, 0.5], duration=0.050, sample_rate=1e3, plot=True):

    #generate vector that represents time
    num_samps = int(duration*sample_rate)
    t = np.arange(num_samps) / sample_rate

    #generate the series
    y = np.zeros([num_samps])
    #generate a random starting point
    y[0] = np.random.randn()
    for k in range(1, num_samps):
        #determine the number of previous time points available
        nw = min(k+1, len(weights))
        #multiply each weight by the previous point in the series
        for j in range(nw):
            y[k] += y[k-(j+1)]*weights[j]

    if plot:
        plt.figure()
        plt.plot(t, y, 'g-', linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

    return t,y

generate_ar_process()
plt.show()
[[code]]

**EXERCISE**: Play around with the coefficients and see what you get. Did the series blow up into infinity? Some basic conditions on how to make the AR process **stable** are outlined in [[@http://davegiles.blogspot.com/2013/06/when-is-autoregressive-model.html|this blog]].


===2. Fourier Transforms and Power Spectrums=== 

There's alot of good material out there to learn about the Fourier Transform:
* [[@http://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/|An Interactive Guide to the Fourier Transform]]
* [[@https://www.youtube.com/watch?v=gZNm7L96pfY&list=PLB24BC7956EE040CD|The Fourier Transform and It's Applications (Stanford Course)]]
* [[@http://math.stackexchange.com/questions/1002/fourier-transform-for-dummies|"Fourier Transform for Dummies" (stackexchange)]]

The Fourier Transform transforms a **real-valued** **function of time** into a **complex-valued** **function of frequency**. The transformed function can then be used to determine the frequencies present in the time series. The [[@http://en.wikipedia.org/wiki/Spectral_density|power spectrum]] is the squared absolute value of the Fourier Transform. It shows how much power exists in the different frequencies present in the time series.

The most straightforward way to show this is to examine the power spectrum of a time series made up of a bunch of sine waves. Let's use the code to generate sine waves from earlier along with some new code for plotting power spectrums:

[[code format="python"]]
def plot_power_spectrum(x, sample_rate=1.0):

    #take the fourier transform of the time series x
    xft = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), d=1.0/sample_rate)
    findex = freq > 0.0

    #square the magnitude of the fourier transform to get
    #the power spectrum
    ps = np.abs(xft)**2

    #make a plot
    plt.figure()
    plt.plot(freq[findex], ps[findex], 'g-')
    plt.ylabel('Power')
    plt.xlabel('Frequency')

t,y = generate_sine_wave(freqs=[2.0, 57.0, 143.0], duration=5.0, sample_rate=1e3)
plot_power_spectrum(y, sample_rate=1e3)
plt.show()
[[code]]

Note that the x-axis of the power spectrum extends up to 500Hz, exactly one half the specified sampling rate of 1000Hz. The frequency that is one half the sampling rate is called the [[@http://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem|Nyquist Frequency]], it is the **maximum frequency that we can analyze**, given our sampling rate.

**EXERCISE**: Check out the power spectrum of the chaotic logistic map.


===3. Auto Correlation and Cross Correlation Functions=== 

The [[@https://www.youtube.com/watch?v=Q9n10mavanQ|Autocorrelation Function]] gives us some sense of how much "memory" there is in a time series. It is a function that shows the **correlation coefficient** of a time series with itself, at various **lags**. The autocorrelation can be used to test whether a function is random, i.e. whether the values between time points are completely independent.

The autocorrelation function can take a while to understand and effectively utilize. Here we'll just go over how to compute it, and take a look at it for a few different time series. Here's the code to compute it:

[[code format="python"]]
def autocorrelation_function(y, lags=range(20)):
    """ Compute the autocorrelation function for the time series y
        at the given lags.
    """

    acf = np.zeros([len(lags)])
    for k,lag in enumerate(lags):
        #compute the correlation coefficient between y and lagged y
        C = np.corrcoef(y[:len(y)-lag], y[lag:])
        acf[k] = C[0, 1]

    return acf,lags
[[code]]

Let's compute it for a few different time series:

[[code format="python"]]
plt.figure()
#plot the ACF of random noise
y = np.random.randn(1000)
plt.subplot(2, 2, 1)
acf,lags = autocorrelation_function(y, lags=range(50))
plt.plot(lags, acf, 'k-', linewidth=2.0)
plt.title('ACF of Random Noise')
plt.axis('tight')

#plot the ACF of a sine wave
t,y = generate_sine_wave(freqs=[2.0, 57.0, 143.0], duration=5.0, sample_rate=1e3, plot=False)
plt.subplot(2, 2, 2)
acf,lags = autocorrelation_function(y, lags=range(50))
plt.plot(lags, acf, 'k-', linewidth=2.0)
plt.title('ACF of Sum of Sine Wave')
plt.axis('tight')

#plot the ACF of the logistic map
logistic_map = lambda x: 3.86*x*(1.0 - x)
y = run(logistic_map, initial_value=0.5, nsteps=500)
acf,lags = autocorrelation_function(y, lags=range(100))
plt.subplot(2, 2, 3)
plt.plot(lags, acf, 'k-', linewidth=2.0)
plt.title('ACF of Logistic Map (r=3.86)')
plt.axis('tight')

#plot the ACF of another logistic map
logistic_map = lambda x: 3.7*x*(1.0 - x)
y = run(logistic_map, initial_value=0.5, nsteps=500)
acf,lags = autocorrelation_function(y, lags=range(100))
plt.subplot(2, 2, 4)
plt.plot(lags, acf, 'k-', linewidth=2.0)
plt.title('ACF of Logistic Map (r=3.7)')
plt.axis('tight')
plt.show()
[[code]]

The [[@http://en.wikipedia.org/wiki/Cross-correlation|cross correlation]] is like the ACF, except it's measured between two different time series. It's a function that gives the correlation coefficient between two time series as a function of the lag between them.


===4. Linear Filters=== 

[[@http://en.wikipedia.org/wiki/Linear_filter|Linear Filters]] are the bread and butter of [[@http://en.wikipedia.org/wiki/Signal_processing|signal processing]]. A signal is a time series with a duration and sample rate, and a linear filter transforms one signal into another signal. Often, signals are linearly filtered to **subtract out frequencies**, such as a [[@http://en.wikipedia.org/wiki/Low-pass_filter|low pass filter]].

Filtering is accomplished by performing a [[@https://www.khanacademy.org/math/differential-equations/laplace-transform/convolution-integral/v/introduction-to-the-convolution|convolution]] between the signal and the filter. Check out [[@http://mathworld.wolfram.com/Convolution.html|this link]] to see an animated illustration about how convolution works. Basically, the filter slides across the signal, and at every time point, the signal is multiplied by the filter to produce one time point of the output signal.

There is a huge amount of resources out there to learn signal processing, such as [[@https://www.coursera.org/course/dsp|this Coursera class]] or [[@http://ocw.mit.edu/resources/res-6-008-digital-signal-processing-spring-2011/|this MIT DSP class]]. You should take those classes to learn more!

We are going to take a simplistic view of linear filters in the context of Machine Learning. In this view, **we want to predict one time series from another time series**. Imagine one time series is called x(t) and the other is called y(t), and we want to predict y(t) from x(t). The most straightforward way to do this is to **predict y(t) as a weighted combination of the recent history of x(t)**:

[[math]]
\hat{y}(t) ~=~ w_1 x(t) + w_2 x(t-1) + w_3 x(t-2) + \cdots + w_n x(t-n)
[[math]]

The w's can be aggregated into a vector **w**, which is the **linear filter** used to produce the prediction yhat(t).

A linear filter that maps an input to an output is the most simple model in [[@http://en.wikipedia.org/wiki/System_identification|system identification]]. In System Identification, we are given an input time series and an output time series, and our task is to find the function that maps the input to the output.

Although the fields of signal processing and systems identification are vast, we can motivate the use of linear filters with a reasonably simple example. Use this code to generate some input and output data for a system:

[[code format="python"]]
def get_system_data(nsteps=500):
    """ Generate data from linearly filtered smoothed Gaussian noise input. """

    #generate random noise input
    input = np.random.randn(nsteps)

    #smooth the noise with a hanning window
    h = np.hanning(30)
    input_smooth = np.convolve(input, h, mode='same')

    #normalize the input so it's between -1 and 1
    input_smooth /= np.abs(input_smooth).max()

    ##generate the output by convolving with some sort of oscillating filter
    the_filter = [1.0, 0.5, 0.0, -0.5, -1.0, -0.25, 0.0, 0.25, 0.0, -0.05, 0.0, 0.05]
    intercept = 0.7
    y = np.convolve(input_smooth, the_filter, mode='same') + intercept

    return input_smooth,y,the_filter,intercept

input,output,the_filter,intercept = get_system_data(nsteps=500)

plt.figure()
plt.plot(input, 'k-', linewidth=2.0)
plt.plot(output, 'r-', linewidth=2.0)
plt.xlabel('Time')
plt.legend(['Input', 'Output'])
plt.axis('tight')
plt.show()
[[code]]

We would like to find a linear filter that maps the time series named "input" to the time series named "output". We can use regression to do this. First we have to pick the number of time points (also called lags) in our filter. Let's choose 12. Let the input sequence be named x(t) and the output sequence y(t). If the length of each signal is T time points, the data matrix looks like this:

[[math]]
X ~=~
\left[ \begin{array}{ccc}
x(1) & \cdots & x(12) \\
x(2) & \cdots & x(13) \\
x(3) & \cdots & x(14) \\
\vdots & \ddots & \vdots \\
x(T-12) & \cdots & x(T)
\end{array} \right]
[[math]]

The target vector looks like this:

[[math]]
\textbf{y} ~=~
\left[
\begin{array}{c}
y(12) \\
\vdots \\
y(T) \\
\end{array} \right]
[[math]]

Note that we started at y(12), because we have 12 lags, so we can't fit the right filter until we have enough input points to convolve our filter with. The weight vector for our regression problem looks like this:

[[math]]
\textbf{w} ~=~
\left[
\begin{array}{c}
w_{12} \\
\vdots \\
w_{1} \\
\end{array} \right]
[[math]]

So to find a linear filter, we just do simple linear regression! Here's some code to help you fit a linear filter between two time series:

[[code format="python"]]
from sklearn.linear_model import Ridge

def fit_linear_filter(input, output, nlags):
    """ Fit the weights of a linear filter with nlags between
        the input time series and the output time series,
    """

    #generate data matrix
    X = list()
    for k in range(nlags, len(output)):
        X.append(input[k-nlags:k])
    X = np.array(X)

    #generate target vector
    y = output[nlags:]

    #do a ridge regression
    rr = Ridge(alpha=1)
    rr.fit(X, y)

    #return the filter weights and the bias
    return rr.coef_[::-1],rr.intercept_
[[code]]

Let's use it:

[[code format="python"]]
#generate the input/output data
input,output,the_filter,the_intercept = get_system_data()

#fit a linear filter to the input/output data
pred_filter,pred_intercept = fit_linear_filter(input, output, 12)

#generate a predicted output from the input using a convolution
pred_output = np.convolve(input, pred_filter, mode='same') + pred_intercept

#compute the correlation coefficient between the predicted and actual output
C = np.corrcoef(output, pred_output)
cc = C[0, 1]

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(input, 'k-', linewidth=2.0)
plt.plot(output, 'r-', linewidth=2.0)
plt.xlabel('Time')
plt.legend(['Input', 'Output'])
plt.axis('tight')

plt.subplot(2, 2, 2)
plt.plot(the_filter, 'bo-', linewidth=2.0)
plt.plot(pred_filter, 'co-', linewidth=2.0)
plt.xlabel('Lag')
plt.legend(['Actual', 'Predicted'])
plt.title('Filters')
plt.axis('tight')

plt.subplot(2, 2, 3)
plt.plot(output, 'r-', linewidth=2.0)
plt.plot(pred_output, 'g-', linewidth=2.0)
plt.legend(['Actual Output', 'Predicted Output'])
plt.xlabel('Time')
plt.axis('tight')
plt.title('cc=%0.2f' % cc)

plt.show()
[[code]]

In real life - you should use k-fold cross validation to determine your linear filter weights, just as in any other regression. You can generate holdout sets by separating the time series into k folds, fitting on k-1 folds, and computing the correlation coefficient on the kth fold. Do that k times, and average your filter weights to get the final filter.
