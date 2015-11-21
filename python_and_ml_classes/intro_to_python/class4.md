=Topics= 
# Review of FizzBuzz
# Review of IsPrime
# Tuples
# Dictionaries
# String parsing exercise

=In-class Projects= 

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">1. Review of FizzBuzz</span>=== 

[[code format="python"]]
# We'll make a loop that goes through all the
#numbers between 1 and 100
for n in range(1, 101):
    # Now we'll check if the number is divisible
    # by both 3 and 5. When using if/elif/elif, each
    # condition is mutually exclusive, so if the first
    # statement is True, the rest will not be evaluated.
    if n % 3 == 0 and n % 5 == 0:
        print("{0}: fizzbuzz".format(n))
    elif n % 3 == 0:
        print("{0}: fizz".format(n))
    elif n % 5 == 0:
        print("{0}: buzz".format(n))
[[code]]

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">2. Review of Is_Prime</span>=== 
[[code format="python"]]
# we'll import all the functions in the math library,
# so we don't have to prefix the functions with "math."
from math import *

def is_prime(some_integer):
    # if some_integer is less than 2, return True
    if some_integer < 2:
        return True

    # first take the square root of some_integer
    # the result of the square root can be a float, so we'll
    # round it up using ceil, and then cast the result
    # to an int
    max_divisor = int(ceil(sqrt(some_integer)))

    # now we'll check if some_integer is divisible by any
    # number between 2 and max_divisor
    for divisor in range(2, max_divisor + 1):
        # check divisibility using the modulus
        if some_integer % divisor == 0:
            # we found a number that some_integer is
            # divisible by! we know now that the some_integer
            # is not prime, so we'll return False and the function
            # will stop running
            return False

    # if we reach this point in the code, the for loop went through every
    # number between 2 and max_divisor, and some_integer was not
    # divisible by any of those numbers. We can safely say that some_integer
    # is prime, so we'll return True
    return True

# now let's test our function. We'll create a list of numbers and check to see if they are prime
numbers = [0, 1, 10, 11, 593, 2, 7, 8, 1913]

for x in numbers:
    is_x_prime = is_prime(x)
    if is_x_prime:
        print("{0} is prime!".format(x))
    else:
        print("{0} is not prime :(".format(x))
[[code]]
=== === 
===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">3. Tuples</span>=== 

<span style="font-size: 12.8000001907349px;">Tuples are groups of numbers, strings, or lists that are immutable, which means they cannot be changed. Tuples are denoted with the ( ) symbols, and they can also be nested.</span>
[[code format="python"]]
a_coordinates = (3, 7)
a_and_b_coordinates = ((3, 7), (4, 5))
[[code]]
<span style="font-size: 12.8000001907349px;">It's also possible to "unpack" the contents of the tuple by setting it equal to the same number of variables as there are values in the tuple.</span>
[[code format="python"]]
x, y = a_coordinates
a_coords, b_coords = a_and_b_coordinates
[[code]]

<span style="font-size: 12.8000001907349px;">**Exercise:** Make a list of x y coordinates using tuples and print out the x and y values for each pair using a for loop.</span>

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">4. Dictionaries</span>=== 

<span style="font-size: 12.8000001907349px;">Dictionaries are another way to organize information in python. We denote a dictionary using the { } symbols, and the information inside the dictionary is organized using keys.</span>
[[code format="python"]]
trees = {'cherry': 1, 'sorgum': 8, 'apple': 2}
[[code]]
The keys can be strings or numbers or a mix of both, and the values can be pretty much anything, including lists. To add something to a dictionary you place the key in brackets and put the value after the equals sign.
[[code format="python"]]
trees['money'] = 0
print(trees)
trees = {'cherry': 1, 'sorgum': 8, 'apple': 2, 'money': 0}
[[code]]
You can delete something from a dictionary using the "del" command.
[[code format="python"]]
del trees['sorgum']
print(trees)
trees = {'cherry': 1, 'apple': 2, 'money': 0}
[[code]]
To find out if a dictionary contains a specific key, you use the "in" function.
[[code format="python"]]
'money' in trees
True
[[code]]
You can also print out the keys, values, and items of a dictionary directly.
[[code format="python"]]
trees.keys()
['cherry', 'apple', 'money']
trees.values()
[1, 2, 0]
trees.items()
[('cherry', 1), ('apple', 2), ('money', 0)]
[[code]]
**<span style="font-size: 12.8000001907349px;">Exercise:</span>**<span style="font-size: 12.8000001907349px;"> Create a dictionary of you and your friends' names and movie preferences.</span>

===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">5. String Parsing Exercise</span>=== 

<span style="font-size: 12.8000001907349px;">We'll use a tool to download a fake DNA sequence from a URL, and the exercise will be to use 4 for loops to count the number of a's, t's, c's, and g's in the file. Here is code to read the data from a url into a string:</span>

[[code format="python"]]
import urllib2
response = urllib2.urlopen('http://python-biophysics-2014.wikispaces.com/file/view/sequence.txt/521523880/sequence.txt')
sequence_text = response.read()
[[code]]


=Homework= 
* http://www.codecademy.com/courses/python-beginner-en-pwmb1/0/1
> 
