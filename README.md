### Simple implementations of stochastic gradient descent algorithms
# Created by Aurélien Lécuyer and Jérémy Pennont
# Applied Mathematics, Polytech Lyon

We are interested here in stochastic gradient descent algorithms, which are used to minimize functions with suitable smoothness properties. For a machine learning problem, we are looking for parameters that minimise a convex loss function, for which computing the exact gradient is computationally intensive, as all the data set is used. The interest of stochastic algorithms is therefore to use at each iteration an approximation of the gradient computed with a randomly selected subset of the data.

Will use a simple machine learning example, multivariate linear regression, to easily test the different algorithms.