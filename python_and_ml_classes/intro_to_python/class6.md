=Topics= 
# Generating Random Arrays
# Matplotlib Histograms and Plots
# Logical Indexing
# More Matplotlib

=In-class Exercises= 

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">1. Generating Random Arrays and Matrices</span>=== 

Random numbers are awesome to play with and essential for science! There are three types of random numbers we'll work with today:
# **Uniform Random Numbers**: numbers that are floats between 0 and 1
# **Gaussian Random Numbers**: numbers that have a [[@http://en.wikipedia.org/wiki/Normal_distribution|Gaussian Distribution]]
# **Random Integers**: whole-number random values

Random numbers are more fun when you can plot them. So we'll provide some quick examples on generating random numbers and then move to the next section that deals with plotting them:
[[code format="python"]]
#generate some uniform random numbers
u = np.random.rand(10)
print "uniform random numbers ="
print u

#generate some Gaussian random numbers
g = np.random.randn(10)
print "Gaussian random numbers ="
print g

#generate some random integers between 0 and 20
x = np.random.randint(0, 20, 10)
print "random integers ="
print x

#generate a 4x4 matrix of uniform random numbers
rmatrix = np.random.rand(4, 4)
print "random matrix ="
print rmatrix
[[code]]

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">2. Matplotlib Histograms and Plots</span>=== 
A //histogram// shows the //distribution// of an array of numbers. Matplotlib allows us to easily create histograms from arrays. If we're writing code from a script, we first have to import numpy and matplotlib. Let's show an example:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt

#generate a bunch of Gaussian random numbers
g = np.random.randn(5000)

#create a figure
plt.figure()

#create a histogram within the figure
plt.hist(g, bins=20)

#add some labels to the figure
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Gaussian Random Numbers")

#show the plots
plt.show()
[[code]]

**EXERCISE**: Make a histogram of uniform random numbers.

When we're working with data, we'll often have to examine the relationship between two variables. To do that, we can plot one variable vs the other. In this example, we'll plot a signal that changes over time:
[[code format="python"]]
#create an array of time points
t = np.arange(0, 5.0, 1e-2)

#create a sine wave at 3 Hz
freq = 3.0
signal = np.sin(2*np.pi*freq*t)

#create a figure
plt.figure()

#plot time vs the signal
plt.plot(t, signal, 'g-')

#label the figure
plt.xlabel("Time (s)")
plt.ylabel("Signal")
plt.title("A sine Wave")

#show the figure
plt.show()
[[code]]

**EXERCISE**: Add gaussian random noise to the sine wave and plot it.

There are many very useful things we can do with matplotlib. Work through [[@http://matplotlib.org/users/pyplot_tutorial.html|this tutorial]] outside of class to learn more. We'll be using it alot in future classes.

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">3. Logical Indexing</span>=== 
Sometimes you might need to //query// an array, which means to find elements in the array that obey certain criteria. We'll accomplish this using //logical indexing//. A //logical index// is a numpy array with values that are boolean, i.e. either "True" or "False". In the following example, we'll generate some uniform random numbers, and //threshold// any values below 0.5 by setting them to 0:

[[code format="python"]]
#generate some uniform random numbers
x = np.random.rand(15)
print "x before thresholding ="
print x

#query the array to find values below the threshold of 0.5
values_to_threshold = x < 0.5

print "the logical index array looks like this ="
print values_to_threshold

#now index into x to examine the values that are below threshold
print "values in x that are below threshold ="
print x[values_to_threshold]

#set the values in x that are below threshold to 0
x[values_to_threshold] = 0.0
print "x after thresholding="
print x
[[code]]
We can combine logical indices using the //bitwise operators//. In the following example, we'll set all the values that are between 0.3 and 0.7 to 0:
[[code format="python"]]
#generate some random uniform numbers
x = np.random.rand(15)
print "x before thresholding ="
print x

#find the values in x that are between 0.3 and 0.7
values_to_threshold = (x >= 0.3) & (x <= 0.7)

#threshold the values of x
x[values_to_threshold] = 0.0
print "x after thresholding ="
print x
[[code]]

**EXERCISE:** Generate an array of uniform random numbers. Use the //bitwise or// "|" to find all the elements of the array that are either less than 0.3 or greater than 0.7, and set them to 0.

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">4. More Matplotlib</span>=== 
What if we want multiple plots in a figure? We can use //subplots// in matplotlib to do that. Let's revisit the random number example, and plot histograms for two different types of random numbers all in one figure. Let's also throw in a noisy sine wave for fun:
[[code format="python"]]
#create some gaussian random numbers
g = np.random.randn(5000)

#create some uniform random numbers
u = np.random.rand(5000)

#create a sine wave
t = np.arange(0.0, 5.0, 1e-2)
freq = 3.0
signal = np.sin(2*np.pi*freq*t)

#create some noise to add to the sine wave
noise = np.random.randn(len(signal))

#create a figure
plt.figure()

#create our first subplot, a histogram of uniform random numbers
plt.subplot(2, 2, 1)
plt.hist(u, bins=20)
plt.title("Uniform Distribution")

#create the second subplot, a histogram of Gaussian random numbers
plt.subplot(2, 2, 2)
plt.hist(g, bins=20)
plt.title("Gaussian Distribution")

#create the third subplot, a noiseless sine wave
plt.subplot(2, 2, 3)
plt.plot(t, signal, 'r-')
plt.title("A sine Wave")

#create the last subplot, a noisy sine wave
plt.subplot(2, 2, 4)
plt.plot(t, signal+noise, 'g-')
plt.title("A Noisy sine Wave")

#show the figure
plt.show()
[[code]]

**EXERCISE:** Use logical indexing and matplotlib to plot a sine wave, where the positive part of the sine wave is red, and the negative part of the sine wave is blue. The way to do this is to superimpose two plots - one that only plots the positive half of the signal, and another plot that shows the negative half of the signal. You can call plt.plot(...) twice in a row to superimpose two plots in the same figure.


=<span style="color: #333333; font-family: arial,helvetica,sans-serif;">Homework</span>= 
* Read through the [[@http://wiki.scipy.org/Tentative_NumPy_Tutorial|Tentative NumPy Tutorial]] and check out the [[@http://docs.scipy.org/doc/numpy-1.8.1/reference/|reference]].
* Check out the matplotlib [[@http://matplotlib.org/users/pyplot_tutorial.html|tutorial]] and [[@http://matplotlib.org/gallery.html|gallery]].
* Check out [[@http://wiki.scipy.org/NumPy_for_Matlab_Users|NumPy for Matlab Users]]
