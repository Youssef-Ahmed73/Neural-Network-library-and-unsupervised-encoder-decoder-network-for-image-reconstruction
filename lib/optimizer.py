import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        """
        layers: list of Layer objects
        Each layer must implement params() → [(param, grad), ...]
        """
        for layer in layers:
            for (param, grad) in layer.params():
                param -= self.lr * grad
                
    def zero_grad(self, layers):
        """Convenience: zero all gradients for each layer."""
        for layer in layers:
            layer.zero_grad()