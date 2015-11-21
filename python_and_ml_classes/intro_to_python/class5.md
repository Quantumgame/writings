= = 
=Topics= 
# Review of String Parsing Exercise
# Numpy Arrays and Matrices

=In-class Projects= 

===1. Review of String Parsing Exercise=== 

**EXERCISE**: Count the number of A's, T's, C's, and G's in a fake DNA sequence.

[[code format="python"]]
#create some fake sequence text
sequence_text = 'ctgggtactcagactccgatcgccgtcgtagggatgggttatccgtacct'

#initialize a dictionary to keep track of each letter count. the key is the
#letter, and the value is the count. we'll be updating the count using a loop
letter_counts = {'a':0, 't':0, 'c':0, 'g':0}

#use a for loop to iterate through each letter in sequence_text. note that
#a string is basically an array of letters, python allows us to loop through
#a string just as if it is a list.
for letter in sequence_text:
    #the iterator variable "letter" changes each iteration. it starts out at
    #"c", then goes to "t", "g", "g", etc. the next line of code finds the
    #count of the letter, and increments it by one
    letter_counts[letter] += 1

#now let's print out the counts of each letter
print 'Letter Counts:'

#we can iterate through the keys and values of a dictionary using the .items()
#function inside of a for loop, as follows:
for letter,count in letter_counts.items():
    print '{0}: {1}'.format(letter, count)
[[code]]


===2. Numpy Arrays and Matrices=== 

In the last class we familiarized ourselves with lists. A //numpy array// is just a list of numbers or letters, but with extra functions that make it very useful for analyzing data. The first step to using numpy is to import it. Then we can create arrays and manipulate them. Let's jump into an example:
[[code format="python"]]
#import the numpy library, and rename it to "np", which
#is shorter and easier to type
import numpy as np

#create an array of 4 numbers
x = np.array([1.2, -3.0, 2.2, 22.7])
print "x ="
print x

print "the sum of the elements in x ="
print x.sum()

print "the mean of x ="
print x.mean()

print "the standard deviation of x ="
print x.std()

print "the absolute value of x ="
print np.abs(x)
[[code]]
We can create two arrays and add them to create a new array. This is called //elementwise// addition. We can also subtract, multiply, and divide numpy arrays elementwise:
[[code format="python"]]
x = np.array([1, -2, 5, 7.6])
y = np.array([0.5, -3, 4, 2.4])
print "x ="
print x
print "y ="
print y

print "x + y ="
print x + y

print "x - y ="
print x - y

print "x / y ="
print x / y

print "x * y ="
print x * y
[[code]]
What if we would like to add 1 to each of the elements of an array? This is called //broadcasting//, and numpy supports it:
[[code format="python"]]
x = np.array([0.5, 8.8, 7.2, -14.4, 22.57, -3.2, -1.1])
print "x before adding 1.0 ="
print x

#add 1 to every element of x
x += 1.0
print "x after adding 1.0 ="
print x
[[code]]
**<span style="line-height: 1.5;">EXERCISE</span>**<span style="line-height: 1.5;">: Find the largest absolute value of x and divide x by it. This is one way to //normalize// the array, so that all of it's values are between -1 and 1.</span>

Remember that the "range" function generates a list of numbers. Numpy as an equivalent function called "arange":
[[code format="python"]]
print "all the integers from 0 to 19:"
print np.arange(20, dtype='int')

print "decimals between 0 and 1, spaced by 0.1:"
print np.arange(0, 1.1, 1e-1)
[[code]]
**EXERCISE**: Generate a list of integers from 0 to 100, spaced by 5.

To //index// or //slice// an array is to look at a subset of elements in the array. Numpy has some very smart indexing tools:
[[code format="python"]]
x = np.array([9.9, 10.2, -1.3, 24, 5.3, -7.2, 8.0])
print "x="
print x

print "the length of x is {0}".format(len(x))

print "the first element of x ="
print x[0]

print "the last element of x ="
print x[-1]

print "the first 3 elements of x ="
print x[:3]

print "the second and third elements of x ="
print x[1:3]

print "the last 3 elements of x ="
print x[:-3]

print "x with all the positions reversed ="
print x[::-1]
[[code]]
**EXERCISE**: Print out the middle 3 elements of x.

A //matrix// is a two-dimensional array of numbers. By two-dimensional, we mean that each number is indexed by both a //row// and a //column//. We can create a matrix by passing a list of lists into a numpy array:
[[code format="python"]]
the_matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print the_matrix
[[code]]

We can index a matrix by it's rows and columns. Here are some examples:
[[code format="python"]]
print "the upper left value of the_matrix ="
print the_matrix[0, 0]

print "the lower right value of the_matrix ="
print the_matrix[-1, -1]

print "the first row of the_matrix ="
print the_matrix[0, :]

print "the first column of the_matrix ="
print the_matrix[:, 0]

print "the middle of the 2nd row of the_matrix ="
print the_matrix[1, [1, 2] ]

print "the first two rows of the_matrix ="
print the_matrix[:2, :]

print "the first two columns of the_matrix ="
print the_matrix[:, :2]
[[code]]
**<span style="line-height: 1.5;">EXERCISE</span>**<span style="line-height: 1.5;">: Print the last column of the_matrix. In another line, print the last two rows of the_matrix.</span>

One more thing - if we want to generate an array or matrix of all ones or zeros, we can do that:
[[code format="python"]]
#create a 5x5 array of zeros
matrix_of_zeros = np.zeros([5, 5])

#create a 3x4 array of ones
matrix_of_ones = np.ones([3, 4])
[[code]]

=Homework= 
* Read through the [[@http://wiki.scipy.org/Tentative_NumPy_Tutorial|Tentative NumPy Tutorial]] and check out the [[@http://docs.scipy.org/doc/numpy-1.8.1/reference/|reference]].
* Check out the matplotlib [[@http://matplotlib.org/users/pyplot_tutorial.html|tutorial]] and [[@http://matplotlib.org/gallery.html|gallery]].
* Check out [[@http://wiki.scipy.org/NumPy_for_Matlab_Users|NumPy for Matlab Users]]
