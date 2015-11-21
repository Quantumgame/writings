=Topics= 
# Installing the development environment
# Running Code
# Some Notes about Floats
# Boolean Variables and Comparison Operators
# The math Package
# Working with Strings
==== ==== 
=In-class Projects= 

NOTE: The following instructions show how to install Enthought Canopy. An alternative to using Enthought is to use [[@http://continuum.io/downloads|Anaconda]] . One package we use later in the class, Biopython, is easier to install in Anaconda.

===1. Installing Enthought Canopy=== 

Mac Users: [[Installing Enthought Canopy on Mac OSX|Follow these instructions]].
Linux Users: [[Installing Enthought Canopy on Ubuntu|Follow these instructions.]]
Windows Users: [[Installing Enthought on Windows|Follow these instructions.]]

===2. Running Some Code=== 

First, open up a Canopy editor (ask if you don't know how, or read [[@http://docs.enthought.com/canopy/quick-start/code_editor.html|this]]). The editor looks something like this:

[[image:blank_editor.png width="470" height="333"]]

The window on the bottom right is a Python //console//, you can use it to execute python code. The first thing we'll do is the first thing everybody is obliged to do for their first programming exercise - make the program print out "hello world!". Type the following code into your console and then press Enter:
[[code format="python"]]
print('Hello World!')
[[code]]
You should see the text "Hello World" below your python code. "print" is a //function// in the Python programming language, that takes a //string// as an //argument// and writes the string to the //output.//

We can also use print to provide us the results of calculations. Try the following code:
[[code format="python"]]
print(4 + 5)
[[code]]
In the code above, the plus sign is an //operator// that sums two //integers// (4 and 5).The print //function// returns the result of applying the plus operator to the two integers. The result is another integer, 9.

Similar to high-school algebra, in programming we often work with symbols that represent values instead of the values themselves. These symbols are called //variables.// For example, let's define a variable named "x" that contains the value 4:
[[code format="python"]]
x = 4
[[code]]

We can print the value of this variable out using the print function:
[[code format="python"]]
print(x)
[[code]]

We can also work with non-integer valued numbers, which are typically called //floats//. Let's play around with them:
[[code format="python"]]
a = 5.2
print(a + 3.5)
[[code]]

**EXERCISE:** define three variables "x", "y", and "z", equal to three different numbers. Print out the sum of those three numbers.


===3. Some Notes about Floats=== 

Whole numbers are called //integers//, decimal-valued numbers are called //floats////.// When you divide two integers in Python, the result is an integer:
[[code format="python"]]
print(8 / 2)
print(8 / 3)
[[code]]
In the example above, 8 / 2 = 4, while 8 / 3 = 2. When dividing integers, Python always throws away the remainder. This operation is also called taking the //floor//. Python will do float arithmetic when at least one of the numbers involved is a float:
[[code format="python"]]
print(8 / 3.0)
print(8.0 / 3)
[[code]]
When we have a variable that is an integer, but want it to be considered a float, we //cast// it using the "float" keyword:
[[code format="python"]]
x = 8
y = 3
print(x / y)
print(x / float(y))
[[code]]
Likewise, we can convert floats to integers by casting them using the "int" keyword:
[[code format="python"]]
x = 8
y = 3.0
print(x / y)
print(x / int(y))
[[code]]
Using //int// is an easy way to take the floor of a float:
[[code format="python"]]
x = 8.999999
print(x)
print(int(x))
[[code]]


===4. Boolean Variables, Logical Operators, and Comparison Operators=== 

A //boolean// variable can take on one of two values - "True" or "False". Let's get to it:
[[code format="python"]]
a = True
b = False
print(a)
print(b)
[[code]]
Boolean variables are super important! They are used in conjunction with [[http://www.pythonforbeginners.com/basics/python-conditional-statements|conditional statements]] to control the //flow of logic// in a program.

[[http://www.tutorialspoint.com/python/logical_operators_example.htm|Logical Operators]] are operators that work with boolean variables. They are absolutely essential for the operation of every computer on the planet! If you want to understand why, read a bit about [[@http://en.wikipedia.org/wiki/Logic_gate|logic gates]] and the [[http://en.wikipedia.org/wiki/Von_Neumann_architecture|Von Neumann Architecture]].

Logical operators can be represented by [[@http://en.wikipedia.org/wiki/Truth_table|truth tables]]. The truth table for the AND logical operator is:
||   || True || False ||
|| True || True || False ||
|| False || False || False ||
Python offers three logical operators: //and//, //or//, and //not//. Let's mess around with them:
[[code format="python"]]
a = True
b = False
print(a and b)
print(a or b)
print(not a)
print( not (a and b) )
[[code]]

[[@http://www.tutorialspoint.com/python/comparison_operators_example.htm|Comparison Operators]] compare the values of two variables and return a boolean variable. They test for equality, less than/greater than, and not equal. Here are some examples:

[[code format="python"]]
print(4 < 5)
print(5 < 4)
print(20 == 20)
print("dog" == "cat")
print("goats" == "goats")
print("goats" != "goat")
[[code]]

===5. The math Package=== 

The Python [[@https://docs.python.org/2/library/math.html|math package]] contains many familiar functions we can use to work with numbers, such as logarithms, exponents, and trigonometric functions. In a few weeks, we'll be using numpy instead of the math package... but it's important to know it exists. To use the math package, we need to //import// the package using the following code:
[[code format="python"]]
from math import *
[[code]]
Then we can use functions like "log", "exp", "sin", and so forth:
[[code format="python"]]
print(e)
print(pi)
print(log(e))
print(sin(pi))
print(cos(pi))
[[code]]

There is a commonly used operator in most programming languages called the //modulus//. The modulus returns the //remainder// when two numbers are divided. Try it out:
[[code format="python"]]
print(4 % 4)
print(4 % 3)
print(4 % 3.5)
[[code]]

**EXERCISE:** Think of a way to use the modulus to determine whether a given integer is even or odd.


===6. Working with Strings=== 

Strings are sets of characters, including whitespace. Strings can be words, sentences, or arbitrary sets of characters:
[[code format="python"]]
first_name = "Jamie"
last_name = "Morgan"
[[code]]

Strings can be combined (also called //concatenated//), by using the plus operator:
[[code format="python"]]
full_name = first_name + last_name
print(full_name)
[[code]]
**EXERCISE:** Modify the assignment of the "full_name" variable so that there is a space between the first name and last name.

When the string variable itself has a double quote character in it, you need to //escape// that quote by using a //backslash//:
[[code format="python"]]
sentence = "I can't wait to see the movie \"Sharknado 2\" in 3D"
print(sentence)
[[code]]

A very special character is the //newline// character. It represents a line break. The newline character is "\n" and can be added into a string. Let's make a haiku:
[[code format="python"]]
sentence2 = "an alligator\ntattered newspaper baby\ntrue non-sequitor"
print(sentence2)
[[code]]

**EXERCISE:** Make your own haiku (5 syllables in first line, 7 in second, 5 in third).

It will very often happen that we want to print out strings that have the values of variables in them. This is where we need to use the //format// function. Let's just dive into an example:
[[code format="python"]]
number_of_beans = 500
seller_name = "Mr. Smith"
sentence3 = "Hello {0}, I would like to buy {1} beans.".format(seller_name, number_of_beans)
print(sentence3)
[[code]]

**EXERCISE:** Read through the "Number Formatting" subsection of [[@http://mkaz.com/2012/10/10/python-string-format/|this page]], print out the value of Pi to 4 decimal places.


=Additional Exercises= 
# <span style="line-height: 1.5;">Start working through [[@http://www.codecademy.com/en/tracks/python|Code Academy]] curriculum.</span>
# Check out [[@http://www.pythonforbeginners.com/basics/string-manipulation-in-python|this tutorial]] to learn more about strings.
# Visit [[@http://stackoverflow.com/questions/tagged/python|StackOverflow]], an awesome site where people collaboratively help eachother answer programming questions.
# Check out [[@http://python.berkeley.edu/|python.berkeley.edu]] a locally-sourced website filled with hand-crafted artisan Python resources.
