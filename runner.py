import time

import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network
import network2
import random
import sys

from matplotlib import cm


model_name = 'mnist_model.npy'
model2_name = 'mnist_model_2.json'

def main():
    if sys.argv[1] == "train":
        epochs = 30
        print(f'info: training model for {epochs} epochs')
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        start = time.time()
        net = network.Network([784, 100, 10])
        net.SGD(training_data, epochs, 10, 3.0, test_data=test_data)
        end = time.time()
        net.save_model(model_name)
        print(f'train time taken: {round(end-start, 2)} s')
    elif sys.argv[1] == "test":
        # if len(sys.argv) < 3:
        #     print(f'error: test file index not specified. argc: {len(sys.argv)}')
        #     return 1

        net = network.Network([784, 30, 10])
        net.load_model(model_name)

        index = random.randint(0, 1000)
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        test_data = list(test_data)
        test_img = test_data[index][0].reshape((28, 28))
        plt.imshow(test_img, cmap=cm.Greys_r)
        plt.savefig('infer.png')
        test_img_mat = test_data[index][0].reshape((784, 1))


        print(f'info: running inference on img index: {index}')
        res = net.feedforward(test_img_mat)
        print(f'res: {res}')
        out = np.argmax(res)
        print(f'out: {out}')


def main2():
    if sys.argv[1] == 'train':  
        epochs = 5
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        start = time.time()
        net = network2.Network([784, 100, 10], cost=network2.CrossEntropyCost)
        net.SGD(training_data, epochs, 10, 0.5,
                lmbda=5.0,
                evaluation_data=validation_data,
                monitor_evaluation_accuracy=True,
                monitor_evaluation_cost=True,
                monitor_training_accuracy=True,
                monitor_training_cost=True)
        end = time.time()
        net.save(model2_name)
        print(f'train time taken: {round(end-start, 2)} s')
    elif sys.argv[1] == 'test':
        net = network2.load(model2_name)
        index = random.randint(0, 1000)
        training_data, validation_data, test_data \
              = mnist_loader.load_data_wrapper()
        test_data = list(test_data)
        test_img = test_data[index][0].reshape((28, 28))
        plt.imshow(test_img, cmap=cm.Greys_r)
        plt.savefig('infer-2.png')
        test_img_mat = test_data[index][0].reshape((784, 1))


        print(f'info: running inference on img index: {index}')
        res = net.feedforward(test_img_mat)
        print(f'res: {res}')
        out = np.argmax(res)
        print(f'out: {out}')
      

if __name__ == "__main__":
    # main()
    main2()
    print("done.")
