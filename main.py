import numpy as np
import pickle
import matplotlib.pyplot as plt
from mlxtend.data import loadlocal_mnist

class secondTask():
    def __init__(self):
        np.random.seed(1)  # заставим numpy выдавать одинаковые набор случайных чисел для каждого запуска программы
        np.set_printoptions(suppress=True)  # выводить числа в формате 0.123 а не 1.23e-1

        self.lr = 0.01  # значени на которое будет домножаться дельта на каждом шаге
        self.batch = 60  # кол-во изображений использованное для обучения на каждом шаге

        # Объявляем веса:
        # В `X` находятся изображения для обучения, а в `y` значения соответственно
        self.X, self.y = loadlocal_mnist(
            images_path="train-images.idx3-ubyte",
            labels_path="train-labels.idx1-ubyte")

        # В `Xt` находятся изображения для тестирования, а в `yt` значения соответственно
        self.Xt, self.yt = loadlocal_mnist(
            images_path="t10k-images.idx3-ubyte",
            labels_path="t10k-labels.idx1-ubyte")

        self.W1 = np.random.uniform(-0.05, 0.05, (784, 16))
        self.W2 = np.random.uniform(-0.05, 0.05, (16, 2))
        self.W3 = np.random.uniform(-0.05, 0.05, (2, 16))
        self.W4 = np.random.uniform(-0.05, 0.05, (16, 784))

        self.b1 = np.zeros((16,))
        self.b2 = np.zeros((2,))
        self.b3 = np.zeros((16,))
        self.b4 = np.zeros((784,))

        self.X = self.X * (1.0 / 255)
        self.Xt = self.Xt * (1.0 / 255)

        self.batchData = np.empty((60, 784))

        self.outputFirstLayer = np.empty((60, 16))
        self.outputSecondLayer = np.empty((60, 2))
        self.outputThirdLayer = np.empty((60, 16))
        self.outputFourthLayer = np.empty((60, 784))

    def showPictures(self, image):
        plt.figure(figsize=(28, 28))
        imgplot = plt.imshow(image.reshape(28, 28))
        plt.show()

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def sigmoid_prime(self, s):
        return s * (1 - s)

    def mse(self):
        return 2 * (self.outputFourthLayer - self.batchData)





    def forward(self):
        self.outputFirstLayer = self.sigmoid(self.batchData.dot(self.W1) + self.b1)
        self.outputSecondLayer = self.sigmoid(self.outputFirstLayer.dot(self.W2) + self.b2)
        self.outputThirdLayer = self.sigmoid(self.outputSecondLayer.dot(self.W3) + self.b3)
        self.outputFourthLayer = self.sigmoid(self.outputThirdLayer.dot(self.W4) + self.b4)

        return self.outputFourthLayer



    def backward(self):

        e = self.mse()


        z7 = e * self.sigmoid_prime(self.outputFourthLayer)
        dW4 = self.outputThirdLayer.T.dot(z7)
        self.W4 -= dW4 * self.lr

        z6 = z7.dot(self.W4.T)

        z5 = z6 * self.sigmoid_prime(self.outputThirdLayer)
        dW3 = self.outputSecondLayer.T.dot(z5)
        self.W3 -= dW3 * self.lr

        z4 = z5.dot(self.W3.T)

        z3 = z4 * self.sigmoid_prime(self.outputSecondLayer)
        dW2 = self.outputFirstLayer.T.dot(z3)
        self.W2 -= dW2 * self.lr

        z2 = z3.dot(self.W2.T)

        z1 = z2 * self.sigmoid_prime(self.outputFirstLayer)
        dW1 = self.batchData.T.dot(z1)
        self.W1 -= dW1 * self.lr


def main():
    network = secondTask()
    o = np.arange(0)
    for epoch in range(10):
        for i in range(0, network.X.shape[0], network.batch):
            network.batchData = network.X[i:i + network.batch]
            o = network.forward()
            network.backward()


    for i in range(10):
        network.batchData = network.Xt
        newOutput = network.forward()
        print(network.yt[0])
        network.showPictures(newOutput[0])



if __name__ == "__main__":
    main()
