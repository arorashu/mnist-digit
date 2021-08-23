"""
The Neural Network class
"""
import numpy as np
import random


class Network:
  def __init__(self, sizes):
    """
    sizes is an array specifying no of neurons in each layer
    e.g. for layers like -> [2, 3, 1]
    init using `net = Network([2, 3, 1])`
    """
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]


  def sigmoid(self, a):
    return 1.0 / (1.0 + np.exp(-a))


  def sigmoid_prime(self, z):
    """Derivative of the sigmoid function."""
    return self.sigmoid(z)*(1-self.sigmoid(z))


  def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y)

  def feedforward(self, a):
    """Return output of a network if 'a' is input""" 
    for b, w in zip(self.biases, self.weights):
      a = self.sigmoid(np.dot(w, a) + b)
    return a


  def backprop(self, x, y):
    """
    Return a tuple "(nabla_b, nabla_w)" representing the gradient for
    cost function C_x. "nalbla_b" and "nabla_w" are layer-by-layer
    lists of numpy arrays, similar to "self.biases" and "self.weights"
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    # feedforward
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = self.sigmoid(z)
      activations.append(activation)

    # backward pass
    delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.

    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = self.sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return nabla_b, nabla_w


  def update_mini_batch(self, mini_batch, eta):
    """
    Update the network's weights and biases by applying sgd using backprop
    to a single mini batch.
    "mini_batch" is a list of tuples "(x,y)" and "eta" is the learning rate.
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    # basic gradient descent
    # theta = theta - alpha * grad
    self.weights = [w-(eta/len(mini_batch))*nw
                    for w, nw, in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb
                   for b, nb, in zip(self.biases, nabla_b)]


  def evaluate(self, test_data):
    """
    Return the number of test inputs for which the NN outputs the correct result
    NN output is whichever neuron in the final layer has highest activation
    """
    test_results = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in test_data]
    return sum((int(x == y)) for (x, y) in test_results)


  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    """
    Train the neural network using Stochastic Gradient Descent.
    the "training_data" is a list of tuples (x,y) representing the
    training inputs and desired outputs.
    If "test_data" is provided, network will be evaluated against
    it at each epoch, and partial progress printed out.
    """
    if test_data:
      test_data = list(test_data)
      n_test = len(test_data)
      # 10,000 x (784x1, 1x1)
    training_data = list(training_data)
    # 50,000 X (784x1,  10x1)
    # (10x1) vector is basically a boolean vector to indicate the digit (0-9)
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [ training_data[k: k+mini_batch_size]
                        for k in range(0, n, mini_batch_size)]
        
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      
      if test_data:
        print('Epoch{0}, {1} / {2}'. format(
              j, self.evaluate(test_data), n_test))
      else:
        print(f'Epoch {j} complete')



  def __str__(self):
    return (f'num_layers: {self.num_layers},\n'
            f'sizes: {self.sizes},\n'
            f'biases: {self.biases},\n'
            f'weights: {self.weights}')


def testNN():
  net = Network([2, 3, 1])
  print(net)
  res = net.feedforward(np.array([[2, 3]]).T)
  print(f'network output: {res}')


class LOG:
  INFO = "info"
  NONE = "none"


LOG_LEVEL = LOG.INFO


def log(string, level):
  if LOG_LEVEL == LOG.INFO:
    print(f'info: {string}')
 


if __name__ == "__main__":
  testNN()

