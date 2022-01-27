# -*- coding: utf-8 -*-
"""
Simple implementations of stochastic gradient descent algorithms
Created by Aurélien Lécuyer and Jérémy Pennont, January 2022
Applied Mathematics, Polytech Lyon


functions to generate data inputs and outputs
loss function (and its gradient) for multivariate linear regressions
classic optimization algorithms:
    Newton, Gradient descent
stochastic optimization algorithms:
    Robbins-Monroe, SGD, SAG, SAGA
"""
import random as rd
import numpy as np
import time
from numpy.linalg import norm 

def Y_model(x,param):
    """
    Exact application of the multiple linear regression formula

    Parameters
    ----------
    x : np.array()
        vector with the p explanatory variables of a data x.
    param : np.array()
        weights vector .

    Returns
    -------
    real
        y. 

    """
    p = len(param)
    y = 0
    for i in range(p-1):
        y = y + x[i] * param[i]
    return y + param[p-1]

def Y_generation(x,param):
    """
    Generate a scalar response y, given explanatory variables x and a weight vector
    A gaussian noise is added to the linear equation
    
    Parameters
    ----------
    x : np.array()
        vector with the p explanatory variables of a data x.
    param : np.array()
        weights vector.

    Returns
    -------
    real
        y. 

    """
    p = len(param)
    y = 0
    for i in range(p-1):
        y = y + x[i] * param[i]
    return y + param[p-1] + np.random.normal(0,0.5)

#fonction de coût, et ses dérivées
def L(w,xij,yi):
    """
    Loss function for multiple linear regression

    Parameters
    ----------
    w : np.array()
        weights vector (size p).
    xij : np.array()
        design matrix (data inputs) (size nxp).
    yi : np.array()
        data outputs (size n).

    Returns
    -------
    real
        loss (sum of squared errors).

    """
    n = len(xij)
    cost = 0
    for i in range(len(xij)):
        ci = (Y_model(xij[i],w)-yi[i])**2
        cost = cost + ci
    return cost/n

def gradL(w,xij,yi):
    """
    Gradient of the loss function L

    Parameters
    ----------
    w : np.array()
        weights vector (size p).
    xij : np.array()
        design matrix (data inputs) (size nxp).
    yi : np.array()
        data outputs (size n).

    Returns
    -------
    np.array()
        gradient (vector of the p derivatives with respect to w_i, i=1,...,p).

    """
    p=len(w)
    dw = np.zeros(p)
    n = len(xij)
    for k in range(p):
        s1 = 0
        for i in range(n):
            s2 = 0
            for j in range(p):
                s2 = s2 + w[j]*xij[i][j]
            s1 = s1 + xij[i][k]*(s2 - yi[i])
        dw[k] = 2*s1
    return dw/n

def gradL_RM(w,xij,yi):
    """
    Approximation of the gradient of L,
    calculated using a single data unit,
    whose index is randomly chosen.

    Parameters
    ----------
    w : np.array()
        weights vector (size p).
    xij : np.array()
        design matrix (data inputs) (size nxp).
    yi : np.array()
        data outputs (size n).

    Returns
    -------
    dw : np.array()
        gradient approximation.

    """
    p=len(w)
    n = len(xij)
    i = rd.randrange(n)
    dw = np.zeros(p)
    for k in range(p):
        s = 0
        for j in range(p):
            s = s + w[j]*xij[i][j]
        dw[k] = 2*xij[i][k]*(s - yi[i])
    return dw

def gradL_SGD(w,i,xij,yi):
    """
    Approximation of the gradient of L,
    calculated using a single data unit xi
    whose index i is given as a parameter

    Parameters
    ----------
    w : np.array()
        weights vector (size p).
    i : integer
        DESCRIPTION.
    xij : np.array()
        design matrix (data inputs) (size nxp).
    yi : np.array()
        data outputs (size n).

    Returns
    -------
    dw : np.array()
        gradient approximation.

    """
    p=len(w)
    dw = np.zeros(p)
    for k in range(p):
        s = 0
        for j in range(p):
            s = s + w[j]*xij[i][j]
        dw[k] = 2*xij[i][k]*(s - yi[i])
    return dw


def hessian_L(w,xij,yi):
    """
    Hessian of the loss function L

    Parameters
    ----------
    w : np.array()
        weights vector (size p).
    xij : np.array()
        design matrix (data inputs) (size nxp).
    yi : np.array()
        data outputs (size n).

    Returns
    -------
    np.array()
        Hessian.

    """
    p=len(w)
    d2w = np.zeros(p)
    n = len(xij)
    for k in range(p):
        s1 = 0
        for i in range(n):
            s2 = 0
            for j in range(p):
                s2 = s2 + xij[i][j]
            s1 = s1 + xij[i][k]*s2
        d2w[k] = 2*s1
    return d2w/n


#algorithmes de résolution

def Newton(f,df,alpha,xij,yi,theta0,epsilon=1e-5,itemax=1000):
    """
    Newton's method

    Parameters
    ----------
    f : function
        function for which will find theta such that f(theta)=alpha
    df : function
        derivative of f
    alpha : np.array()
        alpha for which will find theta such that f(theta)=alpha
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        data outputs (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    epsilon : real, optional
        convergence treshold. The default is 1e-5.
    itemax : np.array(), optional
        maximum number of iterations. The default is 1000.

    Returns
    -------
    theta : real
        estimated theta such that f(theta)=alpha.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """
    p=len(xij[0])
    theta = theta0
    ite=1
    save = [theta]
    while((norm(f(theta,xij,yi)-alpha)>epsilon)and (ite<itemax)):
        d = df(theta,xij,yi)
        for j in range(p):
            theta[j] = theta[j]-(f(theta,xij,yi)[j]-alpha[j])/d[j]
        ite=ite+1
        save.append(theta)
    return theta,ite,save 


def GD(f,df,xij,yi,theta0,epsilon=1e-5,eta=0.01,itemax=1000):
    """
    Gradient descent algorithm
    
    Parameters
    ----------
    f : function
        function to minimize
    df : function
        derivative of f
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        Y vector (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    epsilon : real, optional
        DESCRIPTION. The default is 1e-5.
    eta : real, optional
        learning rate. The default is 0.01.
    itemax : np.array(), optional
        maximum iteration. The default is 10000.

    Returns
    -------
    theta : real
        estimated theta that minimize f.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """
    p=len(xij[0])
    theta = theta0
    ite=1
    save = [theta]
    d=np.ones(p)
    while((norm(d)>epsilon) and (ite<itemax)):
        d = df(theta,xij,yi)
        theta = theta - eta*d
        ite=ite+1
        save.append(theta)
    return theta,ite,save 



#Robbins-Monroe
def RM(f,alpha,xij,yi,theta0,eta=0.01,b=1,itemax=100000):
    """
    Robbins-Monroe algorithm
    
    Parameters
    ----------
    f : function
        function to minimize
    df : function
        derivative of f
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        Y vector (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    epsilon : real, optional
        DESCRIPTION. The default is 1e-5.
    eta : real, optional
        learning rate. The default is 0.01.
    b : np.array(), optional
        power of the convergent learning rate (1/n)**b. The default is 1.
    itemax : np.array(), optional
        maximum iteration. The default is 100000.

    Returns
    -------
    theta : real
        estimated theta that minimize f.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """
    alpha=np.array(alpha)
    theta = theta0
    ite = 1
    save = [theta]
    while(ite<itemax):
        an = (eta/ite)**b
        theta = theta - an * (f(theta,xij,yi)-alpha)
        save.append(theta)
        ite = ite+1
    return theta,ite,save



#descente de gradient stochastique
def SGD(f,df,xij,yi,theta0,eta=0.01,itemax=100000):
    """
    Stochastic gradient descent algorithm
    
    Parameters
    ----------
    f : function
        function to minimize
    df : function
        derivative of f
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        Y vector (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    epsilon : real, optional
        DESCRIPTION. The default is 1e-5.
    eta : real, optional
        learning rate. The default is 0.01.
    itemax : np.array(), optional
        maximum iteration. The default is 100000.

    Returns
    -------
    theta : real
        estimated theta that minimize f.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """
    n = len(xij)
    p=len(xij[0])
    theta = theta0
    ite=1
    save = [theta]
    d=[1]*p
    index = np.arange(n)
    while((ite<itemax)): 
        np.random.shuffle(index)
        dm = np.zeros(p)
        for i  in range(n):
            d = df(theta,index[i],xij,yi)
            theta = theta - eta*d
            ite=ite+1
            save.append(theta)
            dm=dm+d/n
        #print(norm(dm))
    return theta,ite,save 

#stochastic average gradient
def SAG(f,df,xij,yi,theta0,epsilon=1e-5,eta=0.01,itemax=100000):
    """
    Stochastic average gradient algorithm
    
    Parameters
    ----------
    f : function
        function to minimize
    df : function
        derivative of f
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        Y vector (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    epsilon : real, optional
        DESCRIPTION. The default is 1e-5.
    eta : real, optional
        learning rate. The default is 0.01.
    itemax : np.array(), optional
        maximum iteration. The default is 100000.

    Returns
    -------
    theta : real
        estimated theta that minimize f.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """
    n = len(xij)
    p=len(xij[0])
    theta = theta0
    ite=1
    save = [theta]
    d1=np.zeros(p)
    d=np.ones(p)
    y=np.zeros((n,p))
    while((ite<itemax) and (norm(d1-d) >epsilon)):
        i = rd.randrange(n)
        dfi = df(theta,i,xij,yi)
        d = d1
        d1 = d1 + dfi - y[i]
        y[i] = dfi
        theta = theta - (eta/n * d1)
        ite=ite+1
        save.append(theta)
    return theta,ite,save

def SAGA(f,df,xij,yi,theta0,eta=0.001,lambda0=0,itemax=100000):
    """
    SAGA algorithm
    
    Parameters
    ----------
    f : function
        function to minimize
    df : function
        derivative of f
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        Y vector (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    epsilon : real, optional
        DESCRIPTION. The default is 1e-5.
    eta : real, optional
        learning rate. The default is 0.01.
    lambda0 : real, optional
        coefficient for the proximal operator. The default is 0.
    itemax : np.array(), optional
        maximum iteration. The default is 100000.

    Returns
    -------
    theta : real
        estimated theta that minimize f.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """    
    def prox(lambda0,eta,x):
        '''
        Proximal operator
        '''
        return x/(lambda0*eta + 1)

    n = len(xij)
    p=len(xij[0])
    theta = theta0
    save = [theta]
    ite=1
    table=np.zeros((n,p))
    j = rd.randrange(n)
    dfj = df(theta,j,xij,yi)
    while((ite<itemax)):
        j = rd.randrange(n)
        dfj2 = df(theta,j,xij,yi)
        table[j] = dfj2
        omg = theta - eta * (dfj2 - dfj + np.mean(table,axis=0))
        theta = prox(lambda0,eta,omg)
        dfj=dfj2
        ite=ite+1
        save.append(theta)
    return theta,ite,save

def SAGA2(f,df,xij,yi,theta0,eta=0.001,lambda0=0,itemax=100000,print_time=0):
    """
    SAGA algorithm, with the possibility to print remaining time during execution
    
    Parameters
    ----------
    f : function
        function to minimize
    df : function
        derivative of f
    xij : np.array()
        data inputs (size nxp)
    yi : np.array()
        Y vector (size n)
    theta0 : np.array()
        initial guess for theta (size p)
    eta : real, optional
        learning rate. The default is 0.01.
    lambda0 : real, optional
        coefficient for the proximal operator. The default is 0.
    itemax : np.array(), optional
        maximum iteration. The default is 100000.
    print_time : bool, optional
        if true, print remaining execution time and average partial derivative evolution each n iterations

    Returns
    -------
    theta : real
        estimated theta that minimize f.
    ite : integer
        number of iteration.
    save : np.array()
        succesive values of theta.

    """    
    def prox(lambda0,eta,x):
        '''
        Proximal operator
        '''
        return x/(lambda0*eta + 1)

    n = len(xij)
    p=len(xij[0])
    theta = theta0
    save = [theta]
    ite=1
    table=np.zeros((n,p))
    j = rd.randrange(n)
    dfj = df(theta,j,xij,yi)
    start_time = time.time()
    for i in range(round(itemax/n)):
        for j in range(n): 
            j = rd.randrange(n)
            dfj2 = df(theta,j,xij,yi)
            table[j] = dfj2
            omg = theta - eta * (dfj2 - dfj + np.mean(table,axis=0))
            theta = prox(lambda0,eta,omg)
            dfj=dfj2
            ite=ite+1
        ti = time.time() - start_time
        if(print_time):
            print("ite = ", ite, "  -  remaining time = ",round(ti*round(itemax/n)/(i+1)-ti,2),"s  -  average derivative = ",np.mean(table),'\n')
            #print(np.mean(table),'\n')
        save.append(theta)
    return theta,ite,save