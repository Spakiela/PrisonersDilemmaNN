import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X = [1, 2, 3, 5]

X, y = spiral_data(100, 3)


class NeuralNetwork:
    def __init__(self, learning_rate, inputs, neurons):
        self.weights = learning_rate * np.random.randn(inputs, neurons)
        self.bias = np.zeros((1, neurons))
        self.learning_rate = learning_rate

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.bias


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = NeuralNetwork(learning_rate=.1, inputs=2, neurons=5)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)