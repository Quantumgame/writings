=Class 3= 
# Scripts
# Console I/O
# Conditional Statements
# Functions
# Lists and Loops
# String Parsing

=In-Class Projects= 
=== === 
===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">1. Creating and Running Scripts from the Terminal</span>=== 
<span style="font-size: 12.8000001907349px;">This is the part where you choose what kind of text editor you'll use for coding. A simple, widely used, and useful editor these days is [[@http://www.sublimetext.com/|Sublime Text]]. If you want something more powerful, try out [[@http://www.jetbrains.com/pycharm/|PyCharm]] or [[@https://code.google.com/p/spyderlib/|Spyder]]. I suggest starting simple with Sublime Text, building up your Python skills, and then switching to PyCharm.</span>
<span style="font-size: 12.8000001907349px;">Python programs are a collection of files that have a .py extension. Each of these files has components of the program that work together. A //script// is a python file that is meant to be run from the command line. Let's create a simple script. Create a file called "hello.py" in your text editor, and put in the following code:</span>
[[code format="python"]]
print("Hello World wtf lolz")
[[code]]
<span style="font-size: 12.8000001907349px;">From the Terminal, navigate to the directory where you put this file, and then execute the following command:</span>
[[code format="bash"]]
ipython hello.py
[[code]]
<span style="font-size: 12.8000001907349px;">It should print out the hello world text from above. Now you are a master - of running Python code! You can now use IPython for lightweight screwing around, and when it comes time for serious business, we'll put our code in scripts and run it from the command line.</span>
=== === 
===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">2. Console I/O</span>=== 

<span style="font-size: 12.8000001907349px;">"I/O" stands for "Input/Output". Now we will make a program called "echo.py". It's goal in life will be to repeat what is given to it by the user. We already know how to produce output on the screen, using the //print// command. There is another command, the //raw_input// command, which we can use to get input from the user at the terminal. Place the following code into a file called "echo.py":</span>
[[code format="python"]]
input_from_user = raw_input('Please enter the text you want to be repeated: ')
output_string = "This is the text you gave me: {0}".format(input_from_user)
print(output_string)
[[code]]
<span style="font-size: 12.8000001907349px;">Navigate to the directory where echo.py lives, and then run:</span>
[[code format="bash"]]
ipython echo.py
[[code]]
<span style="font-size: 12.8000001907349px;">Did it work? If not - speak up and we'll make it work! We are now well on our way to making programs.</span>
=== === 
===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">5. Conditional Statements</span>=== 

<span style="font-size: 12.8000001907349px;">Remember boolean variables and comparison operators? We will now use them. Alot. Create this simple program, called "pet.py":</span>
[[code format="python"]]
pet_type = raw_input("What kind of pet do you have? ")
if pet_type == "wallaby":
    print("Wow isn't that illegal?")
else:
    print("I do not have a {0}, guess we have nothing to talk about.".format(pet_type))
[[code]]
<span style="font-size: 12.8000001907349px;">The first thing to notice is that we are using an if/else statement. Note the colons - they are important. Also note the indentation - four spaces. Indentation is very important in Python. It does not like if your indentation is not uniform. We will encounter this issue often, so keep an eye out for it.</span>
<span style="font-size: 12.8000001907349px;">What if we want to check for an additional pet? We can use an if/elif/else statement:</span>
[[code format="python"]]
pet_type = raw_input("What kind of pet do you have? ")
if pet_type == "wallaby":
    print("Wow isn't that illegal?")
elif pet_type == "python":
    print("Oh how ironic!")
else:
    print("I do not have a {0}, guess we have nothing to talk about.".format(pet_type))
[[code]]
=== === 
===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">6. Functions</span>=== 

<span style="font-size: 12.8000001907349px;">We already have a bit of experience //using// functions - "print" is a function. A function takes zero or more inputs, and can produce zero or more outputs. For example, here is a function that adds two numbers:</span>
[[code format="python"]]
def add(a, b):
 """ Add two numbers, returns their sum. """
 return a + b
[[code]]
<span style="font-size: 12.8000001907349px;">Note that this function //returns// the sum of it's arguments. We can call the function as follows:</span>
[[code format="python"]]
c = add(5, 6)
[[code]]

<span style="font-size: 12.8000001907349px;">**EXERCISE:** Create a script with the add function in it, and call the function in the script. Run the function from the command line.</span>

<span style="font-size: 12.8000001907349px;">Functions don't always return values, here's an example that just prints the sum to the screen instead of returning it:</span>
[[code format="python"]]
def add(a, b):
 """ Adds two numbers, prints the output, returns nothing. """
 print(a + b)
[[code]]
=== === 
===<span style="color: #333333; font-family: arial,helvetica,sans-serif;">7. Lists and Loops</span>=== 

<span style="font-size: 12.8000001907349px;">A list is a group of objects, whether they be numbers, strings, booleans, or more generally, objects. For now we'll keep it simple and talk about list of numbers. Here are two ways of defining the same list of 5 numbers:</span>
[[code format="python"]]
list1 = [0, 1, 2, 3, 4]
print(list1)
list2 = range(5)
print(list2)
[[code]]
<span style="font-size: 12.8000001907349px;">A //for// //loop// iterates over a list of numbers, for example, this code prints out all the numbers from 1 to 20:</span>
[[code format="python"]]
for k in range(20):
    print(k)
[[code]]
<span style="font-size: 12.8000001907349px;">We'll learn alot more about loops soon, but we know enough now to do two important exercises.</span>

<span style="font-size: 12.8000001907349px;">**EXERCISE:** For all the numbers up to 100, print "fizz" if the number is divisible by 3, "buzz" if the number is divisible by 5, and "fizzbuzz" if the number is divisible by both 3 and 5.</span>

<span style="font-size: 12.8000001907349px;">**EXERCISE:** Create a function named "is_prime" that takes an integer argument, and returns True if the number is prime, False otherwise.</span>

=<span style="color: #333333; font-family: arial,helvetica,sans-serif;">Homework</span>= 
* <span style="font-size: 12.8000001907349px;">Code Academy: http://www.codecademy.com/en/tracks/python</span>
