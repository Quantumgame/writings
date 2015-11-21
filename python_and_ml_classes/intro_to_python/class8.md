==<span style="color: #333333; font-family: arial,helvetica,sans-serif;">**Importing CSV data**</span>== 

Ok, now that we have the basics, let's import some real data! We're going to be using the TradeOffData.csv file, so please move or copy the file to your working directory.

We can look at a sample of the data at the beginning using the function .head(), or at the end using .tail(). By default pandas only shows 5 rows, but you can increase the number by passing a value into the function. For example, .head(20) will show 20 rows.
[[code format="python"]]
# Load the file
data = pd.read_csv('TradeOffData.csv')

# View the first 5 rows
data.head()
 Group Treatment Replicate RelativeFitness
0 BKB     Tube     1         0.869963
1 BKB     Tube     2         1.000363
2 BKB     Tube     3         0.982935
3 BAC     Tube     1         0.810392
4 BAC     Tube     2         0.795107
[[code]]
The read_csv function automatically recognizes the csv column headers, and by default the row index starts at zero.
==<span style="color: #333333; font-family: arial,helvetica,sans-serif;">Slicing data frames by values</span>== 

We can slice dataframes according to the values in the data frame. We can use boolean operators like >=, <=, ==, etc., to select the rows where our conditions are met.
[[code format="python"]]
# Find where 'Replicate' is equal to 1
truth_index = data['Replicate'] == 1
# Return rows of data where truth_index is true
data[truth_index]
# Normally we combine these two steps into one line
data[data['Replicate'] == 1]
[[code]]
Instead of boolean operators we can also pass in a list. For example, let's select only the rows where 'Group' is either BKB or BAC.
[[code format="python"]]
data[data['Group'].isin(['BKB', 'BAC'])]
[[code]]
Sometimes we may want to slice the values of a data frame directly. When the data are strings we can slice each individual string using the .str function.
[[code format="python"]]
# The first letter of each string in the Group column
data['Group'].str[0]
[[code]]
<span style="font-size: 12.8000001907349px;">**Exercise**</span>
<span style="font-size: 12.8000001907349px;">Find all the rows where 'RelativeFitness' >= 1.</span>
**Exercise**
Find all the rows where 'Group' starts with the letter D.

==Statistical Functions== 

Now that you know how to get the data you want, we'll look at how to calculate a few basic statistics about your data. We can use the numpy functions we already learned on data frames like this.
[[code format="python"]]
# What is the total number of Replicates?
data['Replicate'].sum()
# What is the average number of Replicates?
data['Replicate'].mean()
# What is the standard deviation of the number of Replicates?
data['Replicate'].std()
[[code]]
By the way, we are just using sum, mean, and std as examples. You can also apply your own function to a column of a data frame. Here is an example where we apply a plus 2 function and return a new data series.
[[code format="python"]]
def add2(series):
 return series + 2
Replicate2 = add2(data['Replicate'])
print(Replicate2)
[[code]]
Pandas also has a nifty little thing called the describe function which gives you a bunch of statistics about your data, collapsed across all the variables. Of course it only cares about numerical data.
[[code format="python"]]
data.describe()

      Replicate RelativeFitness
count 64.000000   64.000000
mean   2.000000    1.192974
std    0.816497    0.297512
min    1.000000    0.795107
25%    1.000000    0.939284
50%    2.000000    1.000340
75%    3.000000    1.510576
max    3.000000    1.699276
[[code]]
**Exercise**
Find the mean and standard deviation of 'RelativeFitness' for all the trials where 'Replicate' >= 2.

==The groupby Function== 

The statistical functions we saw in the last section only let us look at all our data lumped together, but most of the time we actually want to compare statistical values from groups of data. Fortunately pandas has a great feature called "groupby" that helps you do exactly that.
[[code format="python"]]
# This will group the data by each unique value in 'Treatment'
by_treatment = data.groupby('Treatment')
[[code]]
Note that by_treatment is a groupby object, not a data frame! It just tells pandas how to split up the data frame so you can apply functions to each group separately.
[[code format="python"]]
# you can now describe each treatment group separately
by_treatment.describe()
      Replicate RelativeFitness
Treatment
Dish
count 32.000000  32.000000
 mean  2.031250   1.456359
 std   0.822442   0.184792
 min   1.000000   0.955221
 25%   1.000000   1.429005
 50%   2.000000   1.510884
 75%   3.000000   1.581340
 max   3.000000   1.699276
Tube
count 32.000000  32.000000
 mean  1.968750   0.929589
 std   0.822442   0.050153
 min   1.000000   0.795107
 25%   1.000000   0.915050
 50%   2.000000   0.939089
 75%   3.000000   0.953505
 max   3.000000   1.000363
[[code]]
**Exercise**
Find the mean of each //combination// of 'Group' and 'Treatment'.
(hint: you can pass in a list to the groupby function)

==The agg Function== 

We're going to do one more example of applying functions to groups of data. Here we use the agg function, which takes your group and collapses across all variables, then applies the specified list of functions.
[[code format="python"]]
# Find the sum, mean, standard deviation, and length of RelativeFitness for your groups
by_treatment['RelativeFitness'].agg([np.sum, np.mean, np.std, len])
[[code]]
Again, you can pass in your own functions into the agg function, just make sure that the functions accept a series as input and return a single value as output.

**Exercise:** Create your own function which aggregates many values into a single value, then use agg to apply your new function to your groups.

=<span style="color: #333333; font-family: arial,helvetica,sans-serif;">Homework</span>= 
This is the best tutorial for pandas on the web.
http://pandas.pydata.org/pandas-docs/stable/10min.html
