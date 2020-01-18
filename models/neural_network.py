import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    else:
        return 1 / (1+np.exp(-x))


class Network(object):
    def __init__(self, x, y):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 4)
        self.w3 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        self.y1 = sigmoid(np.dot(self.input, self.w1))
        self.y2 = sigmoid(np.dot(self.y1, self.w2))
        self.y3 = sigmoid(np.dot(self.y2, self.w3))
        return self.y3

    def backward(self):
        error_l3 = 2 * (self.y - self.output) * sigmoid(self.output, derivative=True)
        delta3 = np.dot(self.y2.T, error_l3)
        error_l2 = np.dot(error_l3, self.w3.T)*sigmoid(self.y2, derivative=True)
        delta2 = np.dot(self.y1.T, error_l2)
        error_l1 = np.dot(error_l2, self.w2.T) * sigmoid(self.y1, derivative=True)
        delta1 = np.dot(self.input.T, error_l1)

        self.w1 += delta1
        self.w2 += delta2
        self.w3 += delta3

    def train(self):
        self.output = self.feed_forward()
        self.backward()


x=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0],[1],[1],[0]), dtype=float)

NN = Network(x, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(x))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feed_forward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feed_forward()))))  # mean sum squared loss
        print("\n")

    NN.train()