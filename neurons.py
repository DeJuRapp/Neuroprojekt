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
  _c : np.ndarray
  _sigma : np.ndarray
  old_y : np.ndarray
  old_x : np.ndarray

  def init_random(self, neurons:int, weight_shape: int, min:float, max:float):
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''
    self.num_neurons = neurons
    self._c = np.random.uniform(min, max, (neurons, 1, weight_shape))
    self._sigma = np.full((neurons, 1, weight_shape), np.sqrt(weight_shape) * (max-min) / np.sqrt(2 * neurons))

  def propagate(self, x:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of neurons defined by the array w.
    '''
    self.old_x = x
    exponent:np.ndarray = np.square((x - self._c) * self._sigma)
    exponent = np.sum(exponent, axis=2).reshape(*exponent.shape[:-1], 1)
    self.old_y = np.exp(-exponent)
    return self.old_y.T

  def back_propagate(self, derivatives:np.ndarray, train_rate:float=0.1) -> np.ndarray:
    '''
    Returns the derivatives for x.
    '''
    d_c = 2 * (self.old_x - self._c) * self.old_y * derivatives.T

    d_sigma = -d_c * self._sigma
    d_c = -d_sigma * self._sigma
      
    self._c = self._c - np.sum(train_rate * d_c, axis=1).reshape(self._c.shape)
    self._sigma = self._sigma - np.sum(train_rate * d_sigma, axis=1).reshape(self._sigma.shape)

    return -d_c

  @property
  def c(self) -> np.ndarray:
    return self._c.reshape(self.num_neurons, -1)
