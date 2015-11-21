#we'll import all the functions in the math library,
#so we don't have to prefix the functions with "math."
from math import *

def is_prime(some_integer):
    #if some_integer is less than 2, return True
    if some_integer < 2:
        return True

    #first take the square root of some_integer
    #the result of the square root can be a float, so we'll
    #round it up using ceil, and then cast the result
    #to an int
    max_divisor = int(ceil(sqrt(some_integer)))

    #now we'll check if some_integer is divisible by any
    #number between 2 and max_divisor
    for divisor in range(2, max_divisor+1):
        #check divisibility using the modulus
        if some_integer % divisor == 0:
            #we found a number that some_integer is
            #divisible by! we know now that the some_integer
            #is not prime, so we'll return False and the function
            #will stop running
            return False

    #if we reach this point in the code, the for loop went through every
    #number between 2 and max_divisor, and some_integer was not
    #divisible by any of those numbers. We can safely say that some_integer
    #is prime, so we'll return True
    return True

#now let's test our function. We'll create a list of numbers and check to see if they are prime
numbers = [0, 1, 10, 11, 593, 2, 7, 8, 1913]

for x in numbers:
    is_x_prime = is_prime(x)
    if is_x_prime:
        print("{0} is prime!".format(x))
    else:
        print("{0} is not prime :(".format(x))


