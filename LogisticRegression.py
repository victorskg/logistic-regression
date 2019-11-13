import math
import numpy as np

class LogisticRegression(object):
    def __init__(self, learn_rate, epochs, data_set, inputs):
        self.inputs = inputs
        self.epochs = epochs
        self.data_set = data_set
        self.learn_rate = learn_rate
        self.prepare_data(self.data_set)

    def prepare_data(self, data_set):
        np.random.shuffle(data_set)
        train_size = int(0.8 * len(data_set))
        self.train_set, self.test_set = data_set[:train_size], data_set[train_size:]

    def train(self):
        gradient = 0
        inputs = self.inputs
        self.weights = np.array([0.0 for i in range(len(inputs))])

        for _ in range(self.epochs):    
            for iris in self.train_set:
                x = self._get_inputs(iris, inputs)
                y = int(iris[len(iris)-1])
                gradient += self._descending_gradient(y, x, self.weights)

            self.weights += (self.learn_rate * gradient)


    def test(self):
        hits = 0
        inputs = self.inputs
        for iris in self.test_set:
            value = self.classify(self._get_inputs(iris, inputs), self.weights)
            hits = hits + 1 if value == int(iris[len(iris)-1]) else hits
            #print(self._get_inputs(iris, inputs))
            #print('Predict: {0}, Class: {1}'.format(value, int(iris[len(iris)-1])))
        return (hits / len(self.test_set)) * 100

    @staticmethod
    def classify(inputs, w):
        value = np.dot(inputs, w)
        return 1 if value >= 0 else -1

    @staticmethod
    def _descending_gradient(y, x, w):
        return y / (1 + np.exp(y * np.dot(x, w)))

    @staticmethod
    def _get_inputs(row, inputs):
        return [row[inputs[i]] for i in range(len(inputs))]