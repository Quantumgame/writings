
#We'll make a loop that goes through all the
#numbers between 1 and 100
for n in range(1, 101):
    #Now we'll check if the number is divisible
    #by both 3 and 5. When using if/elif/elif, each
    #condition is mutually exclusive, so if the first
    #statement is True, the rest will not be evaluated.
    if n % 3 == 0 and n % 5 == 0:
        print("{0}: fizzbuzz".format(n))
    elif n % 3 == 0:
        print("{0}: fizz".format(n))
    elif n % 5 == 0:
        print("{0}: buzz".format(n))

