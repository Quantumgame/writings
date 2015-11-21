=Topics= 
# Data Frames

=In-class Exercises= 

==**Data frames**== 

Pandas is a python library for analyzing data that is based on the R statistics language. In Pandas, information is structured into a data frame, with labels for the row and column indices. We create a data frame using the python dictionary syntax. Here is an example:
[[code format="python"]]
import numpy as np
import pandas as pd

df = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20140920'),
                   'C': np.arange(5),
                   'D': 'foo'})
  A       B    C  D
0 1 2014-09-20 0 foo
1 1 2014-09-20 1 foo
2 1 2014-09-20 2 foo
3 1 2014-09-20 3 foo
4 1 2014-09-20 4 foo
[[code]]

The first column here is the row index (by default it is numbers starting at zero), and each of the following columns are labeled with the letters we gave them. We can also slice data frames just like we did with numpy arrays.
[[code format="python"]]
# Select only column 'A'
df['A']
# Select only row 0
df[0:1]
[[code]]

We can also sort data frames in various ways.
[[code format="python"]]
# Sort the columns in descending order
df.sort_index(axis=1, ascending=False)
# Sort the rows by column 'C' in descending order
df.sort(columns='C', ascending=False)
[[code]]
Notice that the sort function does not change the original data frame, but instead returns a sorted copy. This is true of many/most functions in Pandas, and it is an intentional feature to help prevent users from destroying their data. We can apply these functions to the original data frame in either of two ways.
[[code format="python"]]
# Redefine the variable df
df = df.sort(columns='C', ascending=False)
# Set "in place" equal to True
df.sort(columns='C', ascending=False, inplace=True)
[[code]]

<span style="font-size: 12.8000001907349px;">We can also create new columns and delete columns from our data frame.</span>
[[code format="python"]]
# Create a new column 'E'
# Set it equal to 2 times the 'C' column
data['E'] = 2*data['C']

# Delete the 'E' column
data = data.drop('E', axis=1)
[[code]]
<span style="font-size: 12.8000001907349px;">**Exercise**</span>
Create a new data frame with the columns: Name, Birthday, FavoriteColor, Age, Height. Populate the data frame with at least 5 people (real or imagined, it's up to you).
**Exercise**
Use the 'Age' column to sort your data frame from oldest to youngest. Then use the 'Birthday' column to sort from youngest to oldest.


=Homework= 
This is the best tutorial for pandas on the web.
http://pandas.pydata.org/pandas-docs/stable/10min.html
