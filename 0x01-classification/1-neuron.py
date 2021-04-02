#!/usr/bin/env python3

import numpy as np


class Neuron:

    def __init__(self, nx):

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__A = 0
        self.__b = 0
        self.__W = np.random.randn(nx).reshape(1, nx)

        def b(self):
            return self.__b

        def A(self):
            return self.__A

        def W(self):
            return self.__W
