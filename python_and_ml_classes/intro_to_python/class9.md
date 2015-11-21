Most scientific projects revolve around analyzing data. We'll construct a very simple skeleton app here that shows a general project layout.

In the tutorial, we'll name our project the "MyApp". Replace "MyApp" with whatever you would like to name your project. Note that The project name is important! It doubles as the package name. A package is a folder that contains for a bunch of python files, and can imported using the import statement.

So first, we'll create a folder that will contain the source code and data:

[[code format="bash"]]
cd /your/code/directory
mkdir MyApp
[[code]]

Then we'll create two subdirectories, one that contains the source code itself and one that will contain data:

[[code format="bash"]]
cd MyApp

mkdir data

mkdir myapp

cd myapp

touch __init__.py

[[code]]

Python project convention is that the project is enclosed in a folder with the same name, and the package is in a subfolder of that folder, with the same name. The __init__.py file is necessary so that Python recognizes the myapp directory as one that contains source code.

Even a simple data analysis project can be broken down into two parts - code that imports and cleans the data, and code that analyzes and plots the data. To accommodate this, we'll create two source code files for each of the two parts:

[[code format="bash"]]
touch import_data.py
touch analysis.py
[[code]]

Finally, we'll need a python file that ties everything together, importing the data, cleaning it, and then making a plot. We'll call this file "main.py":

[[code format="bash"]]
touch main.py
[[code]]

Now drag the entire MyApp folder into sublime text, or open up main.py in your favorite text editor. Add the following code:

[[code format="python"]]
if __name__ == '__main__':
    print "Hello Analysis!"
[[code]]

The if statement checks to see if main.py is being run as a script from the command line. You can run the script from the Terminal as follows:

[[code format="bash"]]
cd MyApp
python myapp/main.py
[[code]]

Now let's fill in some sample content to see how it all works together. First, we'll put some code that generates a fake dataset in import_data.py:

[[code format="python"]]
import numpy as np
import pandas as pd

def create_random_sequence(num_letters):
 letters = ['A', 'T', 'C', 'G']
 #create an empty list to hold the letters
 sequence = list()

 for k in range(num_letters):
 #generate a random integer from 0 to 3
 r = np.random.randint(0, 4)
 #get the random letter
 random_letter = letters[r]

 #append it to the sequence
 sequence.append(random_letter)

 return sequence

def import_the_data(num_datapoints=100):
 data = dict()
 #generate random sequence
 data['letter'] = create_random_sequence(num_datapoints)
 #generate some random quantities between 0 and 1
 data['quantity'] = np.random.rand(num_datapoints)
 #generate some Gaussian random numbers
 data['declination'] = np.random.randn(num_datapoints)
 #return a pandas dataframe from the dictionary we constructed
 return pd.DataFrame(data)
[[code]]

Then we'll put some plotting code in analysis.py:
[[code format="python"]]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_the_data(data_frame):
 plt.figure()

 plt.subplot(2, 1, 1)
 #plot the quantity histogram
 plt.hist(data_frame["quantity"])
 plt.title('Quantity')

 #plot the declination histogram
 plt.subplot(2, 1, 2)
 plt.hist(data_frame["declination"], color="r")
 plt.title("Declination")
 #show the plots
 plt.show()
[[code]]

Finally, we'll import this code in main.py so that it imports data and plots it when we run it:

[[code format="python"]]
#import the functions for generating data
from myapp.import_data import *
#import the code to make plots
from myapp.analysis import *

#the next if statement checks to make sure the code
#is being run as a script. If main.py is instead being
#imported, then the __name__ variable will not
#equal "__main__", so the code will not be
#executed. This prevents your script code from being
#accidentally run if you happen to import the script!
if __name__ == '__main__':
 df = import_the_data(500)
 plot_the_data(df)
[[code]]
