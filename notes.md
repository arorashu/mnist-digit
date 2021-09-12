# MNSIT classifier notes


## Observations

1. Can improve efficiency by doing forward pass of whole mini-batch in one matrix-multiply step
  1. is that why torch has the parameter of mini batch size?
  2. i.e. has efficient implementation by default

2. Data loader can be more efficient
  1. i.e. do you load and shuffle
  2. or random sample from memory or buffer...


3. What is the loss that you plot?
  1. For each epoch, the loss over all data points in the training data
  2. Question: isn't computing loss over all data points very expensive?
    What is the better way to do it?



## What did I learn?

1. Backpropagation
  1. at a high level, its computing gradients backwards, using calculus chain rule
  2. It's fairly simple to do the computation
  3. From the output loss, you keep going back to neurons in the previous layer
  4. Each neuron knows the gradient for updating it's weights and biases
  5. It is a local computation at the neuron

2. Pytorch basics
  1. What are tensors. Create tensors from np.ndarray and vice versa.
  2. How pytorch does automatic differentiation. module called `torch.autograd`.
    1. You just define the layers and forward function.
    2. It automatically creates the backward function for you.
    3. Internally creates computation graph, and uses simple rules to chain the derivatives.
  3. Has some predefined loss functions
  4. backpropagation is one line using `loss.backward()`
  5. changing weights is one line using `optimizer.step()`

3. Current implementation is very slow on my CPU.

