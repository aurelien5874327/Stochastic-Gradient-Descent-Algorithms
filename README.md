# Stochastic-Gradient-Descent-Algorithms
 Simple implementation of SGD, SAG, SAGA, and other optimization algorithm

We are interested here in stochastic gradient descent algorithms, which are used to minimize functions with suitable smoothness properties. For a machine learning problem, we are looking for parameters that minimise a convex loss function, for which computing the exact gradient is computationally intensive, as all the data set is used. The interest of stochastic algorithms is therefore to use at each iteration an approximation of the gradient computed with a randomly selected subset of the data.

Will use a simple machine learning example, multivariate linear regression, to easily test the different algorithms.

### Multivariate linear regression

Given a data set $ \{y_i, x_{i,1}, x_{i,2}, ..., x_{i,p-1}\}$ of $n$ statistical unit, the model assumes that the relationship between the variable $y_i$ and the $x_i$ vector is given by the relationship :
$$y_i =  \sum_{j=1}^{p-1}x_{i,j}w_j + w_p + \epsilon_i $$
where $w$ is the $p$-dimensional parameter vector and  $\epsilon_i$ is the error term.
We will write $X \in \mathbb{R}^{n\times p}$ the design matrix, such that 
$$\forall i=1,...,n ~~ \forall j=1,...,p-1 ~~~~ X_{i,j}=x_{i,j}$$ $$\forall i = 1,...,n ~~~~ X_{i,p} = 1$$
We can now simply write $Y = Xw + \epsilon$

So, we are looking for the $w$ vector that minimize the following loss function L :
$$ L = \frac{1}{n}\sum_{i=1}^n (y_i - \sum_{j=1}^{p}X_{i,j}w_j)^2 $$

The partial derivative of $L$ with respect to $w_k$ is :
$$ \frac{\partial L}{\partial w_k} = \frac{2}{n} \sum_{i=1}^n X_{i,k} (y_i - \sum_{j=1}^{p}X_{i,j}w_j)$$

To be computed, this sum requires to use the entire data set. That's why we will use an approximation, using only the $i$-th statistical unit randomly chosen :
$$(\frac{\partial L}{\partial w_k})_{approx} = 2 *  X_{i,k} (y_i - \sum_{j=1}^{p}X_{i,j}w_j)$$ 

We can easily verify that :
$$ \mathbb{E}[2 *  X_{i,k} (y_i - \sum_{j=1}^{p}X_{i,j}w_j)]= \frac{\partial L}{\partial w_k}$$ 

This approximation will be used in stochastic gradient descent algorithm.