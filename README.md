# MNIST digit recognizer


Neural network based techniques to recognize digits from handwritten images in the MNIST dataset

## Setup

Create a python environment:

`$ python3 -m venv env`

Activate the environment

`$ source env/bin/activate`

Install the requirements

`$ pip install -r requirements.txt`

## Running Instructions

1. To train the model, and save it

`$ python runner.py train`

The model will be saved as `mnist_model.py`

2. To check the model output

`$ python runner.py test`

This will load the saved model.
It will print the predicted number.
The image for which the prediction was run will be saved in the
project directory as `infer.png`.


## References
1. http://neuralnetworksanddeeplearning.com/chap1.html



