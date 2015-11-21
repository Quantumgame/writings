1. Systems of Linear Equations and Matrix Notation
2. Solving Linear Systems with Gaussian Elimination
3. Linear Independence and Rank
4. The Data Matrix and Linear Models
5. Linear Models and Least Squares

===1. Systems of Linear Equations and Matrix Notation=== 

[[@http://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/lecture-1-the-geometry-of-linear-equations/|Check out this excellent lecture from Gilbert Strang to get some help with the basics]].

A **linear equation** is a linear combination of two or more **variables**:

[[math]]
b = w_1 x + w_2 y
[[math]]

x and y are the variables, w1 and w2 are called **coefficients**, and b is a constant. Geometrically, an equation with two variables can be rearranged into the familiar slope-intercept equation for a **line**:

[[math]]
y = \frac{w_1}{w_2} x + \frac{b}{w_2}
[[math]]

Likewise, we can define a **plane** using a linear equation with three variables:

[[math]]
b = w_1 x + w_2 y + w_3 z
[[math]]

The coefficients are w1, w2, and w3. In general, a linear equation with N variables takes the following form:

[[math]]
b = w_1 x_1 + w_2 x_2 + \cdots + w_N x_N = \sum_{i=1}^N w_i x_i
[[math]]

We can have any number of equations that involve these N variables. For the purposes of introduction, when we have N variables we'll have N equations. Imagine we have the following **system** of linear equations:

[[math]]
2x - y = 1
[[math]]
[[math]]
x + y = 5
[[math]]

**EXERCISE**: solve this system of equations.

We can rewrite this system of equations using **matrix notation**. We construct a matrix of coefficients, aggregate the variables into a vector, and do the same with the constants on the right hand side of each equation:

[[math]]
A = \left[ \begin{array}{cc}
2 & -1 \\
1 & 1 \\
\end{array} \right]
[[math]]

[[math]]
\textbf{v} = \left[ \begin{array}{c}
x \\
y
\end{array} \right]
[[math]]

[[math]]
\textbf{b} = \left[ \begin{array}{c}
1 \\
5
\end{array} \right]
[[math]]

The matrix equation that describes this system gets written like this:

[[math]]
A\textbf{v} = \textbf{b}
[[math]]

There are two different ways of thinking about **matrix-vector multiplication**. The first is the **column view**, where A**v** is interpreted as a linear combination of the columns of A:

[[math]]
A\textbf{v} =
x \left[ \begin{array}{c} 2 \\ 1 \end{array} \right] +
y \left[ \begin{array}{c} -1 \\ 1 \end{array} \right]
[[math]]

Another perspective is the **row view**, where each row of the result of the matrix multiplication is taken independently:

[[math]]
A\textbf{v} =
\left[ \begin{array}{cc}
2x - 1y \\
x + y
\end{array} \right]
[[math]]

We can use the [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html|dot]] function in NumPy to do matrix-vector multiplication. Let's get acquainted with it:

[[code format="python"]]
import numpy as np

A = np.array([ [2, -1], [1, 1] ])
b = np.array([1, 5])
v = np.array([-1, -1])

u = np.dot(A, v)
print 'actual right hand side b: ',v
print 'Av: ',u
[[code]]

**EXERCISE**: We provided the wrong value for **v**! Plug in the solution you obtained by hand for **v** and verify that it produces **b**.

It should be clear that we can multiply a matrix by a vector, but we can also multiply a matrix by another matrix, as long as the number of columns in the first matrix equal the number of rows in the second matrix. Check out [[@https://www.khanacademy.org/math/algebra2/alg2-matrices/matrix-multiplication-alg2/v/multiplying-a-matrix-by-a-matrix|this video]] to learn more. We can use the dot function to multiply two matrices:

[[code format="python"]]
A = np.array([ [2, -1], [1, 1] ])
B = np.array([ [3, 2], [-1, -0.5] ])

print 'AB ='
print np.dot(A, B)
print 'BA ='
print np.dot(B, A)
[[code]]

Note that matrix multiplication is not [[@http://en.wikipedia.org/wiki/Commutative_property|commutative]], so AB does not equal BA!

There is a special matrix called the **Identity Matrix** that has ones along it's **diagonal** and zeros elsewhere. The 3x3 identity matrix looks like this:

[[math]]
I = \left[ \begin{array}{ccc}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{array} \right]
[[math]]

For a given matrix A, sometimes there exists another matrix called the **inverse** that, when multiplied against A, produces the identity matrix. Mathematically that is written like this:

[[math]]
A A^{-1} = I
[[math]]

Sometimes an inverse will exist, and sometimes it will not. We'll revisit this topic shortly. One more important operation is called the **transpose**, which rotates the matrix so columns become rows and rows become columns. The transpose is written like this:

[[math]]
A^T
[[math]]

Try it out with this example:

[[code format="python"]]
A = np.array([ [-3, -1, 4], [0, 4, 0], [5, -9, -2] ])
print 'A ='
print A
print 'A.T ='
print A.T
[[code]]

When the columns of a matrix are orthogonal to each other, we have an [[@http://en.wikipedia.org/wiki/Orthogonal_matrix|orthogonal matrix]]. The cool thing about orthogonal matrices is that their **inverse is equal to their transpose**:

[[math]]
A^{-1} = A^T
[[math]]

NOTE: A good [[@http://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/|Linear Algebra class]] will give you some geometric intuition about systems of equations and their solutions. It would be very advantageous for you to learn more linear algebra by going through the course materials for that class. This is just a brief introduction that provides the essentials we need to do some Machine Learning.


===2. Solving Linear Systems with Gaussian Elimination=== 

There is an algorithmic way of solving systems of linear equations called [[@http://en.wikipedia.org/wiki/Gaussian_elimination|Gaussian Elimination]]. The details of how the algorithm works are pretty straightforward, but we won't go over it here, you should [[@http://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/lecture-2-elimination-with-matrices/|learn it on your own time]]! It can be implemented on a computer, and in fact, NumPy provides the [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html|solve]] function to solve systems of linear equations.

Let's use the function to solve the system from earlier:

[[code format="python"]]
A = np.array([ [2, -1], [1, 1] ])
b = np.array([1, 5])
v = np.linalg.solve(A, b)

print 'Solution to Av=b is v=',v
[[code]]

So how awesome is that? Now you can solve gigantic linear systems with thousands of variables and equations in just a few lines of code. However, there are some things you should know before jumping into that...


===3. Linear Independence and Rank=== 

**Not all linear systems have solutions**. As an exercise, try to solve the following system using NumPy:

[[math]]
x + 2y = 3
[[math]]
[[math]]
4x + 8y = 6
[[math]]

The error you get should say something like this:

[[code]]
numpy.linalg.linalg.LinAlgError: Singular matrix
[[code]]

It says that the matrix A is [[@http://en.wikipedia.org/wiki/Invertible_matrix#singular|singular]]. When a matrix is singular, it does not have an inverse. Having an inverse is important because the solution to a linear system can be rewritten like this:

[[math]]
A\textbf{v} = b
[[math]]
[[math]]
A^{-1} A \textbf{v} = A^{-1} b
[[math]]
[[math]]
I \textbf{v} = A^{-1} b
[[math]]
[[math]]
\textbf{v} = A^{-1} b
[[math]]

So **if there is no inverse, there is no solution to the linear system**. There are many equivalent ways of determining whether a matrix is invertible. We'll focus on the most basic condition, which is that **the columns cannot be linear combinations of each other**. Each column of a matrix must contribute something new. To see why the matrix above is singular, we can rewrite it using the column view:

[[math]]
A\textbf{v} = \left[ \begin{array}{cc}
1 & 2 \\
4 & 8
\end{array} \right]

\left[ \begin{array}{c}
x \\ y
\end{array} \right]

~=~
x \left[ \begin{array}{c} 1 \\ 4 \end{array} \right]
+
y \left[ \begin{array}{c} 2 \\ 8 \end{array} \right]
[[math]]

Clearly [2 8] = 2*[1 4], they columns are linear combinations of each other! Another way of saying this is they are **linearly dependent**. For an NxN matrix to be invertible, all N vectors must be **linearly independent**, they must not be linear combinations of each other.

The **rank** of a matrix is the **number of linearly independent columns**. For an NxN matrix, it is less than or equal to N. A matrix that is **full rank** has a rank of N, and is always invertible (non-singular).

We can use the [[@http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.linalg.matrix_rank.html|matrix_rank]] function in NumPy to determine the rank of a matrix. Here's an example:

[[code format="python"]]
A = np.array([ [2, -1], [1, 1] ])
print 'A='
print A
print 'rank of A=',np.linalg.matrix_rank(A)

B = np.array([ [1, 2], [4, 8] ])
print 'B='
print B
print 'rank of B=',np.linalg.matrix_rank(B)
[[code]]

**EXERCISE**: Come up with a full rank 3x3 matrix, then a 3x3 matrix that has a rank of 2, and then a 3x3 matrix with a rank of 1.

There is so much more we could learn about linear algebra, and that you should learn on your own time. But this class is about Machine Learning, and for the purposes of brevity we must move on!


===4. The Data Matrix=== 

We are now officially back to working with Machine Learning problems. In a typical **supervised learning** problem, our data is comprised of **samples**. Each **sample** is comprised of a bunch of **features** and the measured value of a value that **depends** on those features.

Imagine that we have M features for each data point, and one continuous dependent variable. For the first sample, we can write the features in vector form as:

[[math]]
\textbf{x}_1 = \left[ x_1^1 ~ x_1^2 ~ \cdots ~ x_1^M \right]
[[math]]

The feature vector itself is **bold** and each feature, which is a scalar, is not bold. The subscript on each feature indicates the sample number and the superscript indicates the feature number. We are not raising each x to a power! There are variations on this theme but the idea is the same.

We'll call the continuous dependent variable y and label it's sample number with a subscript as well. Then data with N samples becomes a set that looks like this:

[[math]]
\mathcal{D} ~=~ \{ (\textbf{x}_1, y_1), ~ (\textbf{x}_2, y_2), ~ \cdots ~ (\textbf{x}_N, y_N) \}
[[math]]

We want to **model** each output as a **linear combination** of features:

[[math]]
\hat{y_1} ~=~ w_1 x_1^1 ~ w_2 x_1^2 ~ \cdots ~ w_M x_1^M
[[math]]

This is called a **linear model**. Each feature is multiplied by a **weight** wi. Why the hat over the y1, you say? Because we are trying to **approximate** y1, which is the actual measured output, with a linear combination of features, and yhat1 is the approximation.

When you're running algorithms on your data, often you will first have to aggregate the feature vectors into an NxM matrix, which looks like this:

[[math]]
X = \left[ \begin{array}{cccc}
x_1^1 & x_1^2 & \cdots & x_1^M \\
x_2^1 & x_2^2 & \cdots & x_2^M \\
\vdots & \vdots & \ddots & \vdots \\
x_N^1 & x_N^2 & \cdots & x_N^M
\end{array} \right]
[[math]]

The samples of the dependent (or output) variable are aggregated into a vector:

[[math]]
\textbf{y} = \left[ \begin{array}{c}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{array} \right]
[[math]]

We also aggregate the weights into a vector:
[[math]]
\textbf{w} = \left[ \begin{array}{c}
w_1 \\
w_2 \\
\vdots \\
w_N
\end{array} \right]
[[math]]

The approximation for the entire vector of outputs is then a simple matrix-vector multiplication:

[[math]]
\hat{ \textbf{y} } ~=~ X\textbf{w}
[[math]]



===5. Linear Models and Least Squares=== 

The name of the game when it comes to linear models is this:

[[math]]
\text{find the} ~ \textbf{w} ~ \text{that makes} ~ \hat{\textbf{y}} ~ \text{as close as possible to} ~ \textbf{y}
[[math]]

If the number of datapoints was equal to the number of features, the data matrix would be square, and perhaps we could just solve the following linear system:

[[math]]
X\textbf{w} = \textbf{y}
[[math]]

In fact, let's try that out:

[[code format="python"]]
N = 5
M = N
#create a random data matrix
X = np.random.randn(N, M)
#create a random weight vector
w = np.random.randn(N)

#generate output samples
y = np.dot(X, w)

#solve the linear system and see if the weights match
w_est = np.linalg.solve(X, y)

print 'actual weight vector: ', w
print 'linear system solution weight vector: ', w_est
[[code]]

**EXERCISE**: Add some noise to y and solve the system again. Does it still solve for the correct **w**?

In practice, there are a some reasons we cannot estimate the weights of a linear model by solving the above system:

# The output samples are noisy.
# There are more data points than there are features (N > M), so X is rectangular, and rectangular matrices are always singular in some way.

So what should we do? We can formulate a [[@http://en.wikipedia.org/wiki/Loss_function|cost function]] that judges the **fitness** of a given weight vector **w**. The most commonly used cost function is the **sum of squares error**:

[[math]]
E(\textbf{w}) ~=~ \sum_{i=1}^N \left( \hat{y}_i - y_i \right)^2 ~=~ \sum_{i=1}^N \left( \textbf{x}_i \cdot \textbf{w} - y_i \right)^2
[[math]]

The **optimal** **w** is the **w** that **minimizes the sum of squares error**. How do we minimize the sum of squares error? We use calculus! We will come back to optimization in the context of maximum likelihood and generalized linear models very soon, but for now, know this:

**It is possible to analytically solve for the optimal weight vector w using Calculus.**

When we do that, the optimal solution looks like this:

[[math]]
\textbf{w} ~=~ \left( X^T ~ X \right)^{-1} ~ X^T \textbf{y}
[[math]]

We can use the following code to explore least squares:

[[code format="python"]]
def least_squares_fit(X, y):
 #compute the autocorrelation matrix, which is X.T*X
 autocorr = np.dot(X.T, X)

 #invert the autocorrelation matrix
 autocorr_inv = np.linalg.inv(autocorr)

 #compute the matrix-vector multiplication X.T*y
 crosscorr = np.dot(X.T, y)

 #compute the optimal weights w
 w = np.dot(autocorr_inv, crosscorr)

 return w

def create_fake_data(num_samples, num_features, noise_std=0.0):
 """ Create a matrix of fake data, some random weights, and produce an output data vector y. """

 #create a random data matrix
 X = np.random.randn(num_samples, num_features)

 #create a random weight vector
 w = np.random.randn(num_features)

 #generate output samples
 y = np.dot(X, w)

 #add some random noise to y
 y += np.random.randn(num_samples)*noise_std

 return X,y,w

X,y,w = create_fake_data(100, 5, noise_std=0)
w_est = least_squares_fit(X, y)
print 'True value for w: ',w
print 'Least squares estimate for w: ', w_est
[[code]]

**EXERCISE**: Create some fake data using the code above, and estimate the least squares value for the weights. Compare it with the true value for the weights. Add noise using the noise_std parameter of the create_fake_data function, how does the estimate change when noise increases? When the number of samples decrease? What happens when there are more features than data points?
