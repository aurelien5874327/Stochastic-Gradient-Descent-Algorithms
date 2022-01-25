# -*- coding: utf-8 -*-
"""
Simple implementations of stochastic gradient descent algorithms
Created by Aurélien Lécuyer and Jérémy Pennont
Applied Mathematics, Polytech Lyon
"""

#Import library 
import random as rd
import numpy as np
import matplotlib.pyplot as plt

from functions import Y_generation, L, gradL, gradL_RM, gradL_SGD
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

#change wich line is commented to test the different algorithms
#sometimes the learning rate needs to be adapted
#if n or p are too big the learning rate should be decreased 

#theta,ite,save = Newton(gradL,d2L,p*[0],xij,yi,theta0,epsilon)

#theta,ite,save = GD(L,gradL,xij,yi,theta0,epsilon,0.01)

#theta,ite,save = RM(gradL_RM,p*[0],xij,yi,theta0,epsilon,0.01,0.5)

#theta,ite,save = SGD(L,gradL_SGD,xij,yi,theta0,0.001)

#theta,ite,save = SAG(L,gradL_SGD,xij,yi,theta0,epsilon,0.01)

#theta,ite,save = SAGA(L,gradL_SGD,xij,yi,theta0,0.001,0)

theta,ite,save = SAGA2(L,gradL_SGD,xij,yi,theta0,0.0001,0,print_time=1)


print('number of iterations : ',ite)
print('theta : ',theta)

#plot theta evolution
plt.plot(range(len(save)),save)
plt.xlabel('iteration')
plt.ylabel('theta')
plt.title('Evolution of theta')
plt.show()