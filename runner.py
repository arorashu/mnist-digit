import matplotlib.pyplot as plt
import numpy as np

import mnist_loader
import network
import random
import sys

from matplotlib import cm


model_name = 'mnist_model.npy'


def main():
    if sys.argv[1] == "train":
        epochs = 5
        print(f'info: training model for {epochs} epochs')
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
        net = network.Network([784, 30, 10])
        net.SGD(training_data, epochs, 10, 3.0, test_data=test_data)
        net.save_model(model_name)
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


if __name__ == "__main__":
    main()
    print("done.")
