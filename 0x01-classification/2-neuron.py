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
            e = 2.718281885
            fp = np.matmul(self.W, X)+self.b
            fsig = 1/(1+np.exp(-1*fp))
            slef.__A = fsig
            return self
