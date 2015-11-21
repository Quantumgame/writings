1. The Fake Solar Panel Data
2. Exploring the Dataset
3. Turning Data into a Matrix
4. The Training and Validation Sets
5. Ridge Regression with scikits


===1. The Fake Solar Panel Data=== 

Imagine you are a Data Scientist that works for an up-and-coming solar energy company. They're paying you to determine what factors influence the average energy output of a solar panel.
[[image:sun_data.png width="415" height="272" align="left"]]


They've collected data from many customers in many situations, on **features** like hours of sunlight available for that time of year, the weather, the average temperature that day, and whether the panel was clean or dirty. Also your boss has a crazy hypothesis that the number of rabbits visible in a person's yard somehow affect the energy output of a solar panel, and demanded that data be collected. To the left is a [[@http://en.wikipedia.org/wiki/Graphical_model|graphical model]] that can generate our data.


<span style="line-height: 1.5;">We are going to use </span>**<span style="line-height: 1.5;">linear regression</span>** to predict the energy output from the **features** we have been given. Here is the code to generate the data:

[[code format="python"]]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_solar_data(num_samples=100):

    #sunlight possible in a day is gaussian distributed
    sunlight_mean = 12.5
    sunlight_std = 0.9

    #probability mass function for weather (0 = rain, 1 = cloudy, 2 = sun)
    weather_pmf = np.array([0.25, 0.25, 0.50])

    #cdf for weather
    weather_cdf = np.cumsum(weather_pmf)

    #string names of weather
    weather_names = ['rainy', 'cloudy', 'sunny']

    #effect of weather on temperature
    weather_effect_on_temp = np.array([-5.0, -2.0, 2.0])

    #temperature is gaussian distributed by affected by weather and sun
    temp_mean = 10.0
    temp_std = 3.0

    #the sunlight possible in a day has an linear effect on temp
    temp_sun_slope = 1 / 3.0

    #probability that panel is clean
    clean_p = 0.75

    #poisson rate of rabbits
    rabbits_rate = 10

    #mean and std of energy in Watts
    energy_mean = 200.0
    energy_std = 10.0

    #weight for clean
    clean_w = 20

    #weight for weather
    weather_w = np.array([-30, -20, 30])

    #synthesize the data
    data = {'sun':list(), 'weather':list(), 'temp':list(),
            'clean':list(), 'rabbits':list(), 'energy':list()}

    for k in range(num_samples):

        #generate a sample for the possible hours of sunlight for this day
        sun = np.random.randn()*sunlight_std + sunlight_mean

        #generate a sample for the weather for this day
        weather = np.where(weather_cdf >= np.random.rand())[0].min()
        weather_name = weather_names[weather]

        #generate the temperature in celcius, dependent on sun and weather
        temp = np.random.randn()*temp_std + temp_mean + temp_sun_slope*sun + weather_effect_on_temp[weather]

        #generate sample for clean or not clean
        clean = int(np.random.rand() < clean_p)

        #generate number of rabbits
        rabbits = np.random.poisson(rabbits_rate)

        #the effect of temperature on energy is nonlinear
        temp_effect = np.tanh(-0.1*temp) * 100.0

        #the effect of # of hours of sun on energy
        sun_effect = sun*2

        #the effect of weather on energy
        weather_effect = weather_w[weather]

        #the effect of a clean panel on energy
        clean_effect = clean_w*clean

        #generate a sample for the energy
        energy = np.random.randn()*energy_std + energy_mean + sun_effect + weather_effect + clean_effect + temp_effect

        data['sun'].append(sun)
        data['weather'].append(weather_name)
        data['temp'].append(temp)
        data['clean'].append(clean)
        data['rabbits'].append(rabbits)
        data['energy'].append(energy)

    return pd.DataFrame(data)
[[code]]


===2. Exploring the Dataset=== 

Energy output is our **dependent variable**, the variable that we want to predict from the set of features. First we'll use some basic plotting tools to explore the relationship between energy and the features. Let's look at energy compared to weather, and also whether the panel was clean:

[[code format="python"]]
#generate the data
df = generate_solar_data(num_samples=1000)

#boxplot for weather vs energy
df.boxplot('energy', by='weather')

#boxplot for clean vs energy
df.boxplot('energy', by='clean')
[[code]]

Certainly seems like energy is related to those two! Let's check out the relationship between energy and the continuous variables:

[[code format="python"]]
#generate the data
df = generate_solar_data(num_samples=1000)

plt.figure()
#plot sun vs energy
plt.subplot(1, 3, 1)
plt.plot(df['sun'], df['energy'], 'go')
plt.xlabel('Sun (hours)')
plt.ylabel('Energy (Watts)')

#plot temp vs energy
plt.subplot(1, 3, 2)
plt.plot(df['temp'], df['energy'], 'ro')
plt.xlabel('Temp (degrees C)')
plt.ylabel('Energy (Watts)')

#plot rabbits vs energy
plt.subplot(1, 3, 3)
plt.plot(df['rabbits'], df['energy'], 'bo')
plt.xlabel('# of rabbits')
plt.ylabel('Energy (Watts)')
[[code]]

Hmmph. Not so straightforward! We'll include all the variables as features in our regression, including the discrete ones. The next step is to turn our data into a feature matrix.


===3. Turning Data into a Matrix=== 

In order to use a scikits regression model, we need to create a matrix of features. Each row of the matrix is a sample, and each column is a feature. For continuous values such as temperature, this is pretty straightforward, we just make sure that one of the columns of our data matrix is the temperature. When it comes to a binary variable like whether the solar panel is clean, we can use a 0 for not clean and a 1 for clean, which is a perfectly acceptable way of doing regression on binary data.

When we need to regress on discrete features that represent categories, such as the weather, we need to use [[@http://en.wikipedia.org/wiki/One-hot|one-of-k (one-hot)]] encoding. One-of-k encoding turns a category into a binary vector that can be regressed on. For example, weather can take on one of three values, "rainy", "cloudy", or "sunny". We can represent each category by a 3-bit binary vector like this:

[[math]]
\text{rainy} ~=~ \left[ 1 ~ 0 ~ 0 \right]
[[math]]
[[math]]
\text{cloudy} ~=~ \left[ 0 ~ 1 ~ 0 \right]
[[math]]
[[math]]
\text{sunny} ~=~ \left[ 0 ~ 0 ~ 1 \right]
[[math]]

With one-of-k encoding the weather gets broken up into three binary features. Each feature becomes an separate indicator saying if the weather was rainy or not (first feature), cloudy or not (second feature), and sunny or not (third feature).

Although there is a scikits [[@http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html|OneHotEncoder]] preprocessor that would help us transform categorical variables in a programmatic way, I supplied some code that will directly convert our Pandas DataFrame into a format that scikits is comfortable using, which we'll use to do our regression:

[[code format="python"]]
def one_of_k(features, feature_names=None):
    """ Transforms a numpy array of strings into one-of-k coding, where each
        string is represented by a binary vector.
    """

    if feature_names is None:
        feature_names = list(np.unique(features))

    encoded_features = list()
    for fname in features:
        #get the index of the feature
        findex = feature_names.index(fname)
        #create an empty binary vector
        v = [0.0]*len(feature_names)
        #set the bit for the feature
        v[findex] = 1.0
        #append to the list
        encoded_features.append(v)

    return np.array(encoded_features),feature_names

def data_frame_to_matrix(df, dependent_column, categorical_columns=[]):
    """ Convert a pandas DataFrame to a feature matrix and target vector, for
        easy use within scikits.learn models.

        df: The pandas dataframe
        dependent_column: the name of the column that is the dependent variable
        categorical_columns: a list of column names that are categorical, the
            values of that column are re-encoded into one-of-k binary vectors.

        Returns X,y,col_names: X is a matrix of features, the number of rows
            equals the number of samples, and the number of columns is the number
            of features. y is a vector of dependent variable values. col_names is
            a string name that describes each feature.
    """

    #make a list of continuous valued columns
    cont_cols = [key for key in df.keys() if key not in categorical_columns and key != dependent_column]

    #keep track of feature column names
    col_names = list()
    col_names.extend(cont_cols)

    #convert those columns to a matrix
    X = df.as_matrix(cont_cols)

    #convert the categorical columns
    for ccol in categorical_columns:
        #convert the values to one-of-k binary vectors
        ook,feature_names = one_of_k(df[ccol].values)
        #append the feature names
        col_names.extend(['%s_%s' % (ccol, fname) for fname in feature_names])
        #create a new extended feature matrix
        Xext = np.zeros([X.shape[0], X.shape[1]+len(feature_names)])
        Xext[:, :X.shape[1]] = X
        Xext[:, X.shape[1]:] = ook
        X = Xext

    #create the target vector
    y = df[dependent_column].values

    return X,y,col_names
[[code]]


===4. The Training and Validation Sets=== 

The usual purpose of supervised learning with linear regression is to produce a good **predictor**, a linear model that can take a fresh data point that it has not seen before, and accurately predict the dependent variable from it. The ability to predict on unseen data is called **generalization**. A model that fits the data well but does not generalize has **overfit** the data.

The most straightforward strategy to judge if your model is overfitting is to **only train on part of the data**, and **judge performance on the held out data**, called the **test** data. scikits provides the [[@http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html|train_test_split]] function to split our dataset into training and test sets. We'll use the function to split the data in half, using one half for training and the other half for testing:

[[code format="python"]]
from sklearn.cross_validation import train_test_split

#convert Pandas DataFrame to a feature matrix
X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])

#split into training and test sets
Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=0.5)
[[code]]


===5. Regression with scikits=== 

Let's recap what linear regression is, in the context of this problem. We are trying to predict the energy output of a solar panel, a continuous number that is our **dependent variable**, from a set of **features**. The features in our data are hours of sunlight per day, the weather, the temperature, whether the panel was clean or not, and the number of rabbits. A **linear model** will be constructed that takes a **weighted combination** of these features and tries to predict the energy output. The linear model looks like this:

[[math]]
\hat{y} ~=~ w_1 ~*~ \text{clean} ~+~ w_2 ~*~ \text{rabbits} ~+~ w_3 ~*~ \text{sun} ~+~ w_4 ~*~ \text{temp} ~+~ w_5 ~*~ \text{cloudy} ~+~ w_6 ~*~ \text{rainy} ~+~ w_7 ~*~ \text{sunny} ~+~ \text{intercept}
[[math]]

Regression will help us find the weights and the intercept term. The intercept term is a scalar value that accommodates the mean of the dependent variable. Some linear regression algorithms will give you the option of excluding the intercept - **always include the intercept**!

Linear regression works to **minimize the sum-of-squares cost function**:

[[math]]
E(\textbf{w}) ~=~ \sum_{i=1}^N ~ \left( \hat{y}_i - y_i \right) ^2
[[math]]

The sum-of-squares error function can be derived by maximizing the likelihood of a Gaussian [[@http://en.wikipedia.org/wiki/Generalized_linear_model|Generalized Linear Model]]. We'll come back to this in the future.

Ridge regression minimizes the sum-of-squares cost function, and has an additional term that **minimizes the magnitudes of the weights**:

[[math]]
E(\textbf{w}) ~=~ \sum_{i=1}^N \left( \hat{y}_i - y_i \right) ^2 ~+~ \alpha \sum_{i=1}^N w_i^2
[[math]]

The variable called **alpha** in the cost function determines how far the weights are pushed towards zero. The higher the alpha, the more the weights are pushed to zero. This is a form of [[@http://en.wikipedia.org/wiki/Regularization_%28mathematics%29|regularization]], which helps prevent overfitting and facilitates better generalization (a higher R-squared on the test set).

Without any further ado - let's write some code to fit a regression model on our training data and examine the weights:

[[code format="python"]]
#import the Ridge class from scikits
from sklearn.linear_model import Ridge

#create a Ridge object
rr = Ridge()

#fit the training data
rr.fit(Xtrain, ytrain)

#print out the weights and their names
for weight,cname in zip(rr.coef_, col_names):
 print "{}: {:.6f}".format(cname, weight)
print "Intercept: {:.6f}".format(rr.intercept_)

#print out the R-squared on the test set
r2 = rr.score(Xtest, ytest)

print "R-squared: {:.2f}".format(r2)
[[code]]

The last quantity we printed out is called the R-squared, or [[@http://en.wikipedia.org/wiki/Coefficient_of_determination|coefficient of determination]]. It ranges from 0 to 1 and measures the fraction of variance in our dependent variable captured by our linear model. The higher the R-squared, the better your model. A R-squared of 0 means that your model totally sucks, an R-squared of 1 means that there's probably a bug in your code. Somewhere in-between those is more reasonable. Check out [[@https://www.khanacademy.org/math/probability/regression/regression-correlation/v/r-squared-or-coefficient-of-determination|this video]] for a more detailed explanation! Also do not place too much faith in the R-squared, some people think it's crap. Wikipedia has a relatively [[@http://en.wikipedia.org/wiki/Regression_model_validation|moderate]] perspective.

As an exercise, I would like you to explore how the weights, the test set error, and the R-squared on the test set change with the following parameters:
# The number of samples
# The size of the test set
# The alpha of the Ridge Regression

Here's some code to help you do that:

[[code format="python"]]
def run_full_example(df, ridge_alpha=1.0, test_set_fraction=0.5):

    #convert Pandas DataFrame to a feature matrix
    X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])

    #split into training and test sets
    Xtrain,Xtest,ytrain,ytest = train_test_split(X, y, test_size=test_set_fraction)
    print '# of training samples: {}'.format(len(ytrain))
    print '# of test samples: {}'.format(len(ytest))
    print 'alpha: {:.2f}'.format(ridge_alpha)
    print ''

    #create a Ridge object
    rr = Ridge(alpha=ridge_alpha)

    #fit the training data
    rr.fit(Xtrain, ytrain)

    #print out the weights and their names
    for weight,cname in zip(rr.coef_, col_names):
        print "{}: {:.6f}".format(cname, weight)
    print "Intercept: {:.6f}".format(rr.intercept_)
    print ''

    #compute the prediction on the test set
    ypred = rr.predict(Xtest)

    #compute the sum-of-squares error on the test set, which is
    #proportional to the log likelihood
    sqerr = np.sum((ytest - ypred)**2) / len(ytest)
    print 'Normalized Sum-of-squares Error: {:.3f}'.format(sqerr)

    #compute the sum-of-squares error for a model that is just
    #comprised of the mean on the training set
    sqerr_mean_only = np.sum((ytest - ytrain.mean())**2) / len(ytest)
    print 'Normalized Sum-of-squares Error for mean-only: {:.3f}'.format(sqerr_mean_only)

    #print out the R-squared on the test set
    r2 = rr.score(Xtest, ytest)
    print "R-squared: {:.2f}".format(r2)
    print ''

#use the code to explore different values of alpha
df = generate_solar_data(num_samples=50)

#alpha = 0, no regularization
run_full_example(df, ridge_alpha=0.0, test_set_fraction=0.5)

#alpha = 1, some regularization
run_full_example(df, ridge_alpha=1.0, test_set_fraction=0.5)

#alpha = 100, heavy regularization
run_full_example(df, ridge_alpha=100.0, test_set_fraction=0.5)
[[code]]


===k-Fold Cross Validation=== 

There are some lingering questions with our regression:
# Is there a way to interpret model weights relative to each other?
# How can we estimate the significance of model weights?
# What is an optimal value for the Ridge Regression alpha?

The answer to question #1 is - consider [[@http://en.wikipedia.org/wiki/Standard_score|z-scoring]] your continuous features before regressing on them. Z-scoring a feature means to subtract the mean from each sample and then divide by the standard deviation. This leaves your features having a mean of zero and a standard deviation of 1. Many algorithms will do this automatically without your knowledge - so you might as well do it yourself so you know for sure! Binary features can be left alone, as can one-of-k encoded categorical features.

The answer to #2 and #3 involves using [[@http://en.wikipedia.org/wiki/Cross-validation_%28statistics%29|cross-validation]], specifically k-fold cross validation. k-Fold Cross Validation means to separate our data into k non-overlapping segments, or **folds**. Then we train on k-1 folds, and compute the test error on the held out fold. We do this k times, and average the weights and error metrics across folds to get statistics on them. We'll use scikits [[@http://scikit-learn.org/stable/modules/cross_validation.html#k-fold|KFold]] class to do this. Here's some updated code that does it all:

[[code format="python"]]
def run_cross_validation(df, alphas_to_try=[0.1, 0.5, 1.0, 5.0, 25.0], nfolds=10):
    """ Use k-fold cross validation to fit the weights of the solar panel data, and
        also to determine the optimal ridge parameter.
    """

    #import KFold from scikits
    from sklearn.cross_validation import KFold

    #keep track of the mean and std of the error for each ridge parameter
    mean_error_per_ridge_param = list()
    std_error_per_ridge_param = list()

    #keep track of the mean weights and intercept for each ridge parameter
    weights_per_ridge_param = list()
    intercept_per_ridge_param = list()

    #convert Pandas DataFrame to a feature matrix
    X,y,col_names = data_frame_to_matrix(df, 'energy', ['weather'])

    #run k-fold cross validation for each ridge parameter
    for ridge_alpha in alphas_to_try:

        #keep track of the weights and intercept computed on each fold
        weights = list()
        intercepts = list()

        #keep track of the errors on each fold
        errs = list()

        for train_indices,test_indices in KFold(len(df), n_folds=nfolds):
            #break the data matrix up into training and test sets
            Xtrain, Xtest, ytrain, ytest = X[train_indices], X[test_indices], y[train_indices], y[test_indices]

            #create a Ridge object
            rr = Ridge(alpha=ridge_alpha)

            #fit the training data
            rr.fit(Xtrain, ytrain)

            #record the weights and intercept
            weights.append(rr.coef_)
            intercepts.append(rr.intercept_)

            #compute the prediction on the test set
            ypred = rr.predict(Xtest)

            #compute and record the sum-of-squares error on the test set
            sqerr = np.sum((ytest - ypred)**2) / len(ytest)
            errs.append(sqerr)

        #compute the mean weight and intercept
        weights = np.array(weights)
        mean_weights = weights.mean(axis=0)
        std_weights = weights.std(axis=0, ddof=1)
        intercepts = np.array(intercepts)
        mean_intercept = intercepts.mean()
        std_intercept = intercepts.std(ddof=1)

        #compute the mean and std of the test error
        errs = np.array(errs)
        mean_err = errs.mean()
        std_err = errs.std(ddof=1)

        #print out some information
        print 'ridge_alpha={:.2f}'.format(ridge_alpha)
        print '\t Test error: {:.3f} +/- {:.3f}'.format(mean_err, std_err)
        print '\t Weights:'
        for mean_weight,std_weight,cname in zip(mean_weights, std_weights, col_names):
            print "\t\t{}: {:.3f} +/- {:.3f}".format(cname, mean_weight, std_weight)
        print "\tIntercept: {:.3f} +/- {:.3f}".format(mean_intercept, std_intercept)
        print ''

        #record the mean weight and intercept
        weights_per_ridge_param.append(mean_weights)
        intercept_per_ridge_param.append(mean_intercept)

        #record the errors
        mean_error_per_ridge_param.append(mean_err)
        std_error_per_ridge_param.append(std_err)

    #identify the best ridge param
    best_index = np.argmin(mean_error_per_ridge_param)
    best_alpha = alphas_to_try[best_index]
    best_err = mean_error_per_ridge_param[best_index]
    best_weights = weights_per_ridge_param[best_index]
    best_intercept = intercept_per_ridge_param[best_index]


df = generate_solar_data(num_samples=1000)
run_cross_validation(df)
[[code]]
