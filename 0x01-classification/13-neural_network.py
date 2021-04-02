#!/usr/bin/env python3

import numpy as np
from cgi import log


class NeuralNetwork:

    def __init__(self, nx, nodes):

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(nx).reshape(1, nx)
        self.__b2 = 0
        self.__A2 = 0

        @property
        def b1(self):
            return self.__b1

        @property
        def A1(self):
            return self.__A1

        @property
        def W1(self):
            return self.__W1

        @property
        def b2(self):
            return self.__b2

        @property
        def A2(self):
            return self.__A2

        @property
        def W2(self):
            return self.__W2
        
        def forward_prop(self, X):
   
           self.__A1=1/(1+np.exp(-(X*self.__W1+self.__b1)))
           self.__A2=1/(1+np.exp(-(X*self.__W2+self.__b2)))

           return self.__A1,self.__A2 
        
        def cost(self, Y, A):
            """"""
            n=A.shape[1]
            logLoss=-(1/n)*sum(Y*log(A)+(1-Y)*log(1.0000001-A))

            return logLoss
        
        def evaluate(self, X, Y):
            self.forward_prop(X)
            cost=self.cost( Y,self.__A2)
          
            if self.__A2>=0.5:
                prediction=1
            else:
                prediction=0
            
            return prediction,cost
        
        def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
            n=A1.shape[1]
            dW1=(-2/n)*sum(X*(Y-(self.__W1*X+self.__b1)))
            dW2=(-2/n)*sum(X*(Y-(self.__W2*X+self.__b2)))
            db1=(-2/n)*sum(Y-(self.__W1*X+self.__b1))
            db2=(-2/n)*sum(Y-(self.__W2*X+self.__b2))
           
            self.__W1=self.__W1- alpha*dW1
            self.__W2=self.__W2- alpha*dW2
            self.__b1=self.__b1- alpha*db1
            self.__b2=self.__b2- alpha*db2
