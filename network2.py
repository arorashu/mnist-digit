"""
Network2
The Neural Network class
with improvements such as:
 - cross-entropy cost function
 - regularization
 - better initialization of network weights
"""

#### Libraries
# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np



#### Cost functions
class CrossEntropyCost(object):
  @staticmethod
  def fn(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))


  @staticmethod
  def delta(z, a, y):
    return a-y


class QuadraticCost(object):
  @staticmethod
  def fn(a, y):
    return 0.5*np.linalg.norm(a-y)**2

  @staticmethod
  def delta(z, a, y):
    return (a-y) * sigmoid_prime(z) 


#### Main network class
class Network:
  def __init__(self, sizes, cost=CrossEntropyCost):
    """
    sizes is an array specifying no of neurons in each layer
    e.g. for layers like -> [2, 3, 1]
    init using `net = Network([2, 3, 1])`
    """
    self.num_layers = len(sizes)
    self.sizes = sizes
    self.default_weight_initializer()
    self.cost = cost


  def default_weight_initializer(self):
    """
    better default weight initialization
    weights are gaussian random variables with
    mean = 0 and std_dev = 1/sqrt(no of input connections to neuron)
    """
    self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
    self.weights = [np.random.randn(y, x)/np.sqrt(x)
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]


  def large_weight_initializer(self):
    """
    default weight initialization
    weights are gaussian random variables with
    mean = 0 and std_dev = 1 
    """
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x)
                    for x, y in zip(sizes[:-1], sizes[1:])]



  def cost_derivative(self, output_activations, y):
    """Return the vector of partial derivatives \partial C_x /
    \partial a for the output activations."""
    return (output_activations-y)


  def feedforward(self, a):
    """Return output of a network if 'a' is input""" 
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a) + b)
    return a


  def backprop(self, x, y):
    """
    Return a tuple "(nabla_b, nabla_w)" representing the gradient for
    cost function C_x. "nalbla_b" and "nabla_w" are layer-by-layer
    lists of numpy arrays, similar to "self.biases" and "self.weights"
    This function implements the 4 equations of backprop
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
      activation = sigmoid(z)
      activations.append(activation)

    # backward pass
    delta = (self.cost).delta(zs[-1], activations[-1], y)
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
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

    return nabla_b, nabla_w


  def update_mini_batch(self, mini_batch, eta, lmbda, n):
    """
    Update the network's weights and biases by applying sgd using backprop
    to a single mini batch.
    "mini_batch" is a list of tuples "(x,y)" and "eta" is the learning rate.
    "lmbda" is the regularization parameter, and
    "n" is the total size of training set
    """
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    # basic gradient descent
    # theta = theta - alpha * grad
    self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
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


  def SGD(self, training_data, epochs, mini_batch_size, eta, 
          lmbda = 0.0,
          evaluation_data=None,
          monitor_evaluation_cost=False,
          monitor_evaluation_accuracy=False,
          monitor_training_cost=False,
          monitor_training_accuracy=False):
    """
    Train the neural network using mini-batch Stochastic Gradient Descent.
    the "training_data" is a list of tuples (x,y) representing the
    training inputs and desired outputs.
    The method also accepts "evaluation_data", usually either test_data
    or evaluation_data. We can monitor the cost and accuracy on either
    the training data or evaluation data by setting the
    appropriate flags.
    """

    if evaluation_data:
      evaluation_data = list(evaluation_data)
      n_data = len(evaluation_data)
      # 10,000 x (784x1, 1x1)
    training_data = list(training_data)
    # 50,000 X (784x1,  10x1)
    # (10x1) vector is basically a boolean vector to indicate the digit (0-9)
    n = len(training_data)

    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []

    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k: k+mini_batch_size]
                      for k in range(0, n, mini_batch_size)]
        
      for mini_batch in mini_batches:
        self.update_mini_batch(
            mini_batch, eta, lmbda, len(training_data))
      print(f'info: Epoch {j} training complete')
      
      if monitor_training_cost:
        cost = self.total_cost(training_data, lmbda)
        training_cost.append(cost)
        print(f'info: Cost on training data: {cost}')
      if monitor_training_cost:
        accuracy = self.accuracy(training_data, convert=True)
        training_accuracy.append(accuracy)
        print(f'info: Accuracy on training data: {accuracy} / {n}')

      if monitor_evaluation_cost:
        cost = self.total_cost(evaluation_data,lmbda, convert=True)
        evaluation_cost.append(cost)
        print(f'info: Cost on evaluation data: {cost}')
      if monitor_evaluation_accuracy:
        accuracy = self.accuracy(evaluation_data)
        evaluation_accuracy.append(accuracy)
        print(f'info: Accuracy on evaluation data: {accuracy} / {n_data}')
        
    return (evaluation_cost, evaluation_accuracy,
         training_cost, training_accuracy)


  def total_cost(self, data, lmbda, convert=False):
    """
    Return the total cost for data set "data"
    The flag "convert" should be False if data set is the 
    tranining data (usual case), and True if it is validation
    or test data.
    """
    cost = 0.0
    for x, y in data:
      a = self.feedforward(x)
      if convert:
        y = vectorized_result(y)
      cost += self.cost.fn(a, y) / len(data)
    cost += 0.5*(lmbda/len(data))*sum(
        np.linalg.norm(w)**2 for w in self.weights)
    return cost


  def accuracy(self, data, convert=False):
    """
    """
    if convert:
      results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                for (x, y) in data]
    else:
      results = [(np.argmax(self.feedforward(x)), y)
                for (x, y) in data]
    return sum(int(x==y) for (x,y) in results)


  def __str__(self):
    return (f'num_layers: {self.num_layers},\n'
            f'sizes: {self.sizes},\n'
            f'biases: {self.biases},\n'
            f'weights: {self.weights}')


  def save(self, filename):
    """save the nn model to file "filename". """
    data = {'sizes': self.sizes,
            'weights': [w.tolist() for w in  self.weights],
            'biases': [b.tolist() for b in self.biases],
            'cost': str(self.cost.__name__)}
    with open(filename, 'w') as f:
      json.dump(data, f)
    print(f'info: saved modle to {filename}')


  def save_model(self, filename):
    """
    save the model to a file on disk
    """
    model = np.asanyarray([self.num_layers, self.sizes, self.weights, self.biases], dtype=object)
    np.save(filename, model, allow_pickle=True)


  def load_model(self, filename):
    """
    load the neural network model and weights from filename
    """
    [self.num_layers, self.sizes, self.weights, self.biases] = np.load(
        filename, allow_pickle=True)


#### Loading a Network
def load(filename):
  """
  Load a nn from the file "filename". Return an instance of Network
  """
  with open(filename, 'r') as f:
    data = json.load(f)
  
  cost = getattr(sys.modules[__name__], data["cost"])
  net = Network(data["sizes"], cost=cost)
  net.weights = [np.array(w) for w in data["weights"]]
  net.biases = [np.array(b) for b in data["biases"]]

  return net



#### Miscellaneous functions

def vectorized_result(j):
  """
  return a 10-dim unit vector with 1.0 in the j'th position
  and zeroes elsewhere. 
  """
  e = np.zeros((10, 1))
  e[j] = 1.0
  return e


def sigmoid(a):
  return 1.0 / (1.0 + np.exp(-a))


def sigmoid_prime(z):
  """Derivative of the sigmoid function."""
  return sigmoid(z)*(1-sigmoid(z))


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

