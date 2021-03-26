#!/usr/bin/env python3
import numpy as np


class Neuron:

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise TypeError("nx must be a positive integer")

        self.b = 0
        self.W = np.random.randn(nx).reshape(1, nx)
        self.A = 0

        @property
        def b(self):
            return self.__b

        @property
        def A(self):
            return self.__A

        @property
        def W(self):
            return self.__W

        def forward_prop(self, X):
            y = np.matmul(self.W, X)+self.b
            z = 1/(1+np.exp(-1*y))
            slef.__A = z
            return self.__A

        def cost(self, Y, A):
            n = Y.shape[1]
            cost = - (1 / n) * np.sum(np.multiply(Y, np.log(A)) + np.multiply(1 - Y, np.log(1.0000001 - A)))
            return cost

        def evaluate(self, X, Y):

            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            label_value = np.where(self.__A >= 0.5, 1, 0)
            return label_value, cost

        def gradient_descent(self, X, Y, A, alpha=0.05):
           
            dz = A - Y
            dw = np.matmul(X, dz.T) / A.shape[1]
            db = np.sum(dz) / A.shape[1]
            self.__W = self.__W - alpha * dw.T
            self.__b = self.__b - alpha * db
            
