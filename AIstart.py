import numpy as np


X = [1, 2, 3, 4]


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


layer1 = NeuralNetwork(learning_rate=.1, inputs=4, neurons=5)
layer2 = NeuralNetwork(learning_rate=.1, inputs=5, neurons=2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
