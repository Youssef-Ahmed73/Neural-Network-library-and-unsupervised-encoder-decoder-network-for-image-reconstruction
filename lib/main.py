# example_xor.py
import numpy as np
from layers import Dense
from activations import Tanh, Sigmoid, ReLU
from network import NeuralNetwork
from losses import MSE
from optimizer import SGD

# dataset
X = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=float)
Y = np.array([[-1],[1],[1],[-1]], dtype=float)

# model: 2 -> 4 -> 1
layers = [
    Dense(2, 4, activation=ReLU()),
    Dense(4, 1, activation=Tanh())
]

model = NeuralNetwork(layers, loss_function=MSE())
opt = SGD(lr=0.5)

model.train(X, Y, opt, n_epochs=5000, batch_size=4, verbose=True)

print("predictions:", model.predict(X))
