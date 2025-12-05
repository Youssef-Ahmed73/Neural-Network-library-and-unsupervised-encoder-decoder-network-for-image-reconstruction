import numpy as np

class MSE:
    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def grad(self, y_true, y_pred):
        batch = y_true.shape[0]
        return (2.0 * (y_pred - y_true)) / batch
