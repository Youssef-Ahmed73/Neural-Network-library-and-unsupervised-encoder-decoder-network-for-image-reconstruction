# -*- coding: utf-8 -*-

import numpy as np

class Layer:
    """
    Base class for all layers.

    Subclasses should implement:
      - forward(self, X): returns output Y
      - backward(self, dY): returns dX

    Optional:
      - params(self): return iterable of (param_array, grad_array)
      - zero_grad(self): convenience to zero all grads
    """
    def forward(self, X):
        raise NotImplementedError("forward must be implemented by the subclass")

    def backward(self, dY):
        raise NotImplementedError("backward must be implemented by the subclass")

    def params(self):
        # By default no parameters (e.g. ReLU). Subclasses with weights override this.
        return []

    def zero_grad(self):
        # Zero all gradients for optimizer convenience
        for p, g in self.params():
            if g is not None:
                g[...] = 0.0


import numpy as np
from layers import Layer

class Dense(Layer):
    def __init__(self, in_features, out_features, activation=None):
        # Xavier initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2 / in_features)
        self.b = np.zeros((1, out_features))

        self.activation = activation

        # Cache
        self.A_prev = None
        self.Z = None
        self.A = None

        # Gradients
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        """
        Forward pass
        A_prev: input from previous layer (or input data)
        """
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b

        if self.activation is None:
            self.A = self.Z
        else:
            self.A = self.activation.forward(self.Z)

        return self.A

    def backward(self, dL_dA):
        """
        Backward pass
        """
        if self.activation is None:
            dL_dZ = dL_dA
        else:
            dL_dZ = self.activation.backward(dL_dA)

        # Compute gradients
        self.dW = self.A_prev.T @ dL_dZ
        self.db = np.sum(dL_dZ, axis=0, keepdims=True)


        # Gradient to pass back
        dL_dA_prev = dL_dZ @ self.W.T

        return dL_dA_prev

    def params(self):
        return [
            (self.W, self.dW),
            (self.b, self.db)
        ]
