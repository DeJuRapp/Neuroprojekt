from typing import Tuple, List
import numpy as np

class Neuron():
  '''
  The axis order for neurons should always be (neuron, batch, *weights)
  '''
  def init_zeros(self, neurons:int, kernel_dims: Tuple[int]):
    '''
    Returns:
      A (n, 1, kernel_dims) numpy array containing zeros.
    '''

  def init_random(self, neurons:int, weight_shape: np.ndarray, min:float, max:float):
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''

  def propagate(self, x:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of neurons defined by the array w.
    '''

  def back_propagate(self, derivatives:np.ndarray, train:bool=True, train_rate:float=0.1) -> np.ndarray:
    '''
    Returns the derivatives for w and x.
    '''

  def adapt(self, train_rate:float):
    '''
    Adapts the weights according to the derivatives.

    Returns the adapted weights.
    '''

class RBF(Neuron):
  '''
  The neuron implements y = exp(-||x - c||^2 / (2 * b^2))
  '''
  num_neurons: int
  c : np.ndarray
  b : np.ndarray
  old_x : np.ndarray
  old_y : np.ndarray
  old_norm: np.ndarray
  d_c : np.ndarray
  d_b :np.ndarray

  def init_zeros(self, neurons:int, weight_shape: Tuple[int]):
    '''
    Returns:
      A (n, 1, *weight_shape) numpy array containing zeros.
    '''
    self.num_neurons = neurons
    self.c = np.zeros((neurons, 1, *weight_shape))
    self.b = np.ones((neurons, 1))

  def init_random(self, neurons:int, weight_shape: Tuple[int], min:float, max:float):
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''
    self.num_neurons = neurons
    self.c = np.random.uniform(min, max, (neurons, 1, *weight_shape))
    self.b = np.ones((neurons, 1))

  def propagate(self, x:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of neurons defined by the array w.
    '''
    self.old_x = x
    denominator = 2 * np.square(self.b)
    self.old_norm = np.sum(np.square(x.reshape(x.shape[0], x.shape[1], -1) - self.c.reshape(self.num_neurons, 1, -1)), axis=2)
    self.old_y = np.exp(-self.old_norm / denominator).reshape(x.shape[1], self.c.shape[0])
    return self.old_y

  def back_propagate(self, derivatives:np.ndarray, train:bool=True, train_rate:float=0.1) -> np.ndarray:
    '''
    Returns the derivatives for x.
    '''
    self.d_c = self.old_x.reshape(self.old_x.shape[0], self.old_x.shape[1], -1) - self.c.reshape(self.num_neurons, 1, -1)
    self.d_c = self.d_c / np.square(self.b.reshape(*self.b.shape, 1))
    self.d_c *= self.old_y.reshape(self.num_neurons, self.old_x.shape[1], 1)
    self.d_c *= derivatives.T

    self.d_b = self.old_norm / np.power(self.b, 3)
    self.d_b *= self.old_y.T
    self.d_b *= derivatives.T[:,:,0]

    if train:
      self.adapt(train_rate)

    return -self.d_c

  def adapt(self, train_rate:float):
    '''
    Adapts the weights according to the derivatives.

    Returns the adapted weights.
    '''
    self.c = self.c - np.sum(train_rate * self.d_c, axis=1).reshape(self.d_c.shape[0], 1, -1)
    self.b = self.b - np.sum(train_rate * self.d_b, axis=1).reshape(self.d_b.shape[0], 1)
