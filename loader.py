"""
deserialize the mnist pickle and view photos
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import pickle


if __name__ == "__main__":
  
  with open('data/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

  train_x, train_y = train_set
  # print(f'train_x size: {len(train_x)}')
  plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
  # plt.show()
  plt.savefig('a.png')





