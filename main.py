# -*- coding: utf-8 -*-
"""
Simple implementations of stochastic gradient descent algorithms
Created by Aurélien Lécuyer and Jérémy Pennont, January 2022
Applied Mathematics, Polytech Lyon
"""

#Import library 
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import norm 

from functions import Y_generation, gradL, hessian_L, gradL_RM, gradL_SGD
from functions import Newton, RM, GD, SGD, SAG, SAGA, SAGA2


#generating n statistical units
#weights vector w is size p
#design matrix is randomly generated, exept for X[i][p] = 1 for all i = 1,...,n
#Y = Xw  with X de taille n*p 
#the goal will be to find back w, given X and Y
n = 1000
p = 10
xij = np.ones((n,p))

for i in range(n):
    for j in range(p-1):
        xij[i][j] = 10*rd.random()-5
   
w = []
#the p weights are randomly chosen in [-3,3]
for i in range(p):
    if(rd.random()<0.1): #some weights are set to 0
        w.append(0)
    else:
        w.append(6*rd.random()-3)

#the y vector is calculated according to the design matrix and the weight vector
yi = []
for i in range(n):
    yi.append(Y_generation(xij[i],w))
    
#initial guess
theta0 = np.zeros(p)

epsilon=1e-5

#set time
time0 = time.time()

#change wich line is commented to test the different algorithms
#sometimes the learning rate needs to be adapted
#if n or p are too big the learning rate should be decreased 
#for Robbins-Monro or SAGA  it can be necessary to increase itemax if it has not completely converged
#it the convergence is too slow, it is sometimes possible to increase the learning rate

#theta,ite,save = Newton(gradL,hessian_L,p*[0],xij,yi,theta0,epsilon)

#theta,ite,save = GD(gradL,xij,yi,theta0,epsilon,eta=0.01)

#theta,ite,save = RM(gradL_RM,p*[0],xij,yi,theta0,eta=0.01,b=0.5,itemax=1000)

#theta,ite,save = SGD(gradL_SGD,xij,yi,theta0,eta=0.0001)

#theta,ite,save = SAG(gradL_SGD,xij,yi,theta0,epsilon,eta=0.001)

#theta,ite,save = SAGA(gradL_SGD,xij,yi,theta0,eta=0.001,lambda0=0,itemax=50000)

theta,ite,save = SAGA2(gradL_SGD,xij,yi,theta0,eta=0.001,lambda0=0,itemax=50000,print_time=1)

t2 = time.time() - time0

print('number of iterations : ',ite)
print('exectution time : ',t2)
print('1/p * ||theta-w|| : ', norm(theta-w))
print('theta : ',theta)

#plot theta evolution
plt.plot(range(len(save)),save)
plt.xlabel('iteration')
plt.ylabel('theta')
plt.title('Evolution of theta')
plt.show()