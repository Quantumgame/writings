===Outline=== 

1. Points in 1, 2, and 3 Dimensions
2. Linear Combinations

===1. Points in 1, 2, and 3 Dimensions=== 

A point in space is called a **vector**, it has a **magnitude** and a **direction**. I will use the words "point" and "vector" interchangeably. A one-dimensional point requires two things to describe it. The first is a number that indicates it's magnitude, and the second is a sign that's positive or negative that indicates it's direction. The magnitude and direction tell us where the point lies along a straight line that spans from negative infinity to positive infinity. This infinitely long line is the **space** where one dimensional vectors live. We can add, subtract, and multiply 1D points, and the result is another 1D point.

The space that 1D points live in is called the **space of real numbers**, it has a special symbol:

[[math]]
\mathbb{R} ~=~ (-\infty, \infty) ~=~ \text{the set of all numbers from} ~ -\infty ~ \text{to} ~ \infty
[[math]]

When a point x is one dimensional, we can indicate that mathematically by writing:

[[math]]
x \in \mathbb{R}
[[math]]

We need two numbers to describe a point in two-dimensional space, an x,y location. Say we're thinking about a point with an x-coordinate of -1 and y-coordinate of 5. In linear algebra we'll call that point **x** and write it like this:

[[math]]
\textbf{x} ~=~
\left[ \begin{array}{c}
-1 \\
5 \\
\end{array} \right]
[[math]]

That point lives on the two-dimensional [[@http://en.wikipedia.org/wiki/Plane_%28geometry%29|plane]]. That plane also has a special mathematical symbol:

[[math]]
\mathbb{R}^2 ~=~ \text{All possible 2D points up to infinity in both the x and y directions}
[[math]]

To indicate that a point **x** is two dimensional we can write:

[[math]]
\textbf{x} \in \mathbb{R}^2
[[math]]

Like 1D points, we can add and subtract points on the 2D plane. This is done in a straightforward way:

[[math]]
\left[ \begin{array}{c}
a \\
b \\
\end{array} \right]
~+~
\left[ \begin{array}{c}
c \\
d \\
\end{array} \right]

~=~
\left[ \begin{array}{c}
a+c \\
b+d \\
\end{array} \right]
[[math]]

Subtraction is as straightforward:

[[math]]
\left[ \begin{array}{c}
a \\
b \\
\end{array} \right]
~-~
\left[ \begin{array}{c}
c \\
d \\
\end{array} \right]

~=~
\left[ \begin{array}{c}
a-c \\
b-d \\
\end{array} \right]
[[math]]

Adding and subtracting arrays of numbers in this way is called **elementwise** addition and subtraction.

We can also **multiply by a scalar**, which means to take a 2D point and multiply it by a 1D point:

[[math]]
w
\left[ \begin{array}{c}
a \\
b \\
\end{array} \right]

~=~
\left[ \begin{array}{c}
wa \\
wb \\
\end{array} \right]
[[math]]

Let's write some code to better understand addition, subtraction, and scalar multiplication for 2D points. This function will give you an empty figure that you can use to plot points in IPython:

[[code format="python"]]
def create_blank_figure2d():
    plt.figure()
    plt.axhline(0.0, c='k')
    plt.axvline(0.0, c='k')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

def plot_point2d(x):
    assert len(x) == 2, "plot_point only works with 2D points"
    plot(x[0], x[1], 'o')
[[code]]

Run the following code line by line in IPython to explore geometrically what it means to add and subtract points:

[[code format="python"]]
create_blank_figure2d()

v1 = np.array([-1, 5])
v2 = np.array([-2, 3])

plot_point2d(v1)
plot_point2d(v2)
plot_point2d(v1 + v2)
[[code]]

**EXERCISE**: Plot some 2D points, their sums, differences. Take a vector and multiply it by a ton of scalars that range from -1 to 1. What happens?

All the same stuff about elementwise addition, subtract, and scalar multiplication applies to 3D points. Use this code and repeat the above exercise for 3D points:

[[code format="python"]]
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_blank_figure3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylabel('Z')

def plot_point3d(x):
    assert len(x) == 3, "plot_point3d only works with 3d points"
    ax = plt.gca()
    ax.scatter(x[0], x[1], x[2], 'o')
    plt.draw()
[[code]]


===2. Linear Combinations=== 

A **linear combination** is when we multiply vectors by scalars and add them up:

[[math]]
\textbf{v} ~=~

c_1
\left[ \begin{array}{c}
w \\
x \\
\end{array} \right]

~+~

c_2
\left[ \begin{array}{c}
y \\
z \\
\end{array} \right]

~=~

\left[ \begin{array}{c}
c_1 w + c_2 y \\
c_1 x + c_2 z \\
\end{array} \right]
[[math]]

They're kind of a big deal! Why? Well, if we choose the vectors right, we can represent **every possible vector** in two-dimensions from linear combinations of those two vectors. One type of the "right" vectors are vectors that are **orthogonal**. If two vectors are orthogonal, they are geometrically **perpendicular**.

There is another very useful definition of orthogonal that we'll use, but first we need to understand the [[@http://en.wikipedia.org/wiki/Dot_product|dot product]]. The dot product is a different way of multiplying two vectors. It's defined like this:

[[math]]
\textbf{u} ~=~

\left[ \begin{array}{c}
a \\
b \\
\end{array} \right]
[[math]]

[[math]]
\textbf{v} ~=~
\left[ \begin{array}{c}
c \\
d \\
\end{array} \right]
[[math]]

[[math]]
\textbf{u} \cdot \textbf{v} ~=~
ac + bd
[[math]]

Two vectors are **orthogonal** when their **dot product is zero**. In two dimensions, the most basic example of orthogonal vectors are these:

[[math]]
\textbf{u} ~=~
\left[ \begin{array}{c}
1 \\
0 \\
\end{array} \right]
[[math]]

[[math]]
\textbf{v} ~=~
\left[ \begin{array}{c}
0 \\
1 \\
\end{array} \right]
[[math]]

**EXERCISE**: Prove that the vectors **u** and **v** are orthogonal. Come up with more examples of orthogonal vectors in two dimensions. Come up with an example of three orthogonal three-dimensional vectors. Feel free to use a computer, NumPy supplies the [[@http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html|dot]] function that computes the dot product between two vectors.

The entire two dimensional plane can be generated two orthogonal vectors. In the three dimensional plane, we can use three orthogonal vectors. These vectors form a [[@http://en.wikipedia.org/wiki/Basis_%28linear_algebra%29|basis]] in two or three dimensional space, respectively. A basis is a set of vectors whose linear combinations can generate an entire space.

[[code format="python"]]
def plot_linear_combination2d(v1, v2, coef_min=-5.0, coef_max=5.0, num_points=30, marker_size=15.0):
    #create a blank figure
    create_blank_figure2d()

    #compute the dot product
    dp = np.dot(v1, v2)
    plt.title('dot product={:.6f}'.format(dp))

    coefs = np.linspace(coef_min, coef_max, 30)

    #plot all linear combinations of vectors
    for c1 in coefs:
        for c2 in coefs:
            plt.plot(c1*v1[0] + c2*v2[0], c1*v1[1] + c2*v2[1], 'k.', ms=marker_size)

    #plot the line defined by scaling each vector
    clrs = ['b', 'r']
    for c in coefs:
        plt.plot(c*v1[0], c*v1[1], '.', c=clrs[0], ms=marker_size)
        plt.plot(c*v2[0], c*v2[1], '.', c=clrs[1], ms=marker_size)

v1 = np.array([1.0, 0.5])
v2 = np.array([0.0, 1.0])
plot_linear_combination2d(v1, v2, coef_min=-5.0, coef_max=5.0)
[[code]]

**EXERCISE**: Play around with different values for the 2d vectors, explore how "space-filling" their linear combinations appear to be. Do it in 3D as well, here's some code to do it:

[[code format="python"]]
def plot_linear_combination3d(v1, v2, v3, coef_min=-5.0, coef_max=5.0, num_points=15, marker_size=15.0):

    #create a blank figure
    create_blank_figure3d()

    coefs = np.linspace(coef_min, coef_max, num_points)

    #compute all linear combinations of vectors
    all_points = list()
    for c1 in coefs:
        for c2 in coefs:
            for c3 in coefs:
                all_points.append([c1*v1[0] + c2*v2[0] + c3*v3[0],
                                   c1*v1[1] + c2*v2[1] + c3*v2[1],
                                   c1*v1[2] + c2*v2[2] + c3*v3[2]])
    all_points = np.array(all_points)

    #plot all the points
    ax = plt.gca()
    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c='k', s=marker_size, alpha=0.5)

    #plot the line defined by scaling each vector
    clrs = ['b', 'r', 'g']
    coefs_dense = np.linspace(coef_min, coef_max, 100)
    for c in coefs_dense:
        ax.scatter(c*v1[0], c*v1[1], c*v1[2], '.', c=clrs[0], s=marker_size)
        ax.scatter(c*v2[0], c*v2[1], c*v2[2], '.', c=clrs[1], s=marker_size)
        ax.scatter(c*v3[0], c*v3[1], c*v3[2], '.', c=clrs[2], s=marker_size)
[[code]]

Orthogonality of vectors will guarantee that all linear combinations will fill a 2D or 3D space, but we **don't need orthogonality**. All we need is for the vectors to be [[@http://en.wikipedia.org/wiki/Linear_independence|linearly independent]], which we will talk about in the next class.
