import numpy as np
np.random.seed(42)
import random

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.historyOfLearning = []

    def saveHistoryOfLearning(self, error):
        self.historyOfLearning.append(error)
        
    def showParams(self):
        print("number of layers: ", self.num_layers)
        print("size of network: ", self.sizes)
        print("baises: ", "\n", self.biases)
        print("weights: ", "\n", self.weights)

    def showSizeOfParams(self):
        print("number of layers: ", self.num_layers)
        print("size of network: ", self.sizes)
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            print("the size of biase for {0} layer: {1}".format(i+2, b.shape))
            print("the size of weight for {0} layer: {1}".format(i+2, w.shape))

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            # activation layer a' = sigmoid( Wx + b )
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # training_data is a list of tuples of (X, y)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # iterate through data following the given epochs
        for j in xrange(epochs):
            # shuffle the training data
            random.shuffle(training_data)
            # create mini-batch training dataset.
            """
            [i for i in range(0, 100, 20)] => [0, 20, 40, 60, 80]
            this  means the index of mini-batch
            """
            mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                result = float(self.evaluate(test_data))/float(n_test)
                print "Epoch {0}: {1}".format(j, result)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        # initialise b,w with empty list
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        # initialise b, w with empty list: size in this case is [first layer: 30, second layer: 10]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # z = Wx + b
            z = np.dot(w, activation) + b
            zs.append(z)
            # sigmoid(z)
            activation = sigmoid_vec(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives C_x w.r.t a for the output activations
        our case, cost func is C = (1/2)(\sum||output - target||^2)
        Hence, nabla_C = (output - target)
        """
        self.saveHistoryOfLearning(output_activations - y)
        return (output_activations - y)

if __name__ == '__main__':
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return sigmoid(z) * (1 - sigmoid(z))

    # numpy vectorise: https://qiita.com/3x8tacorice/items/3cc5399e18a7e3f9db86
    sigmoid_vec = np.vectorize(sigmoid)
    sigmoid_prime_vec = np.vectorize(sigmoid_prime)


    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = Network([784, 30, 20, 10])

    print(net.SGD(training_data, 1, 10, 3.0, test_data=test_data))
    net.showSizeOfParams()
