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
  _b : np.ndarray
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
    self._c = np.zeros((neurons, 1, *weight_shape))
    self._b = np.ones((neurons, 1, 1))

  def init_random(self, neurons:int, weight_shape: Tuple[int], min:float, max:float):
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''
    self.num_neurons = neurons
    self._c = np.random.uniform(min, max, (neurons, 1, *weight_shape))
    self._b = np.ones((neurons, 1, 1))

  def propagate(self, x:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of neurons defined by the array w.
    '''
    self.old_x = x
    denominator = 2 * np.square(self._b)
    self.old_norm = np.sum(np.square(x - self._c), axis=2)
    self.old_norm = self.old_norm.reshape(*self.old_norm.shape, 1)
    self.old_y = np.exp(-self.old_norm / denominator).T
    return self.old_y

  def back_propagate(self, derivatives:np.ndarray, train:bool=True, train_rate:float=0.1) -> np.ndarray:
    '''
    Returns the derivatives for x.
    '''
    self.d_c = self.old_x - self._c
    self.d_c = self.d_c / np.square(self._b)
    self.d_c *= self.old_y.T
    self.d_c *= derivatives.T

    self.d_b = self.old_norm / np.power(self._b, 3)
    self.d_b *= self.old_y.T
    self.d_b *= derivatives.T

    if train:
      self.adapt(train_rate)

    return -self.d_c

  def adapt(self, train_rate:float):
    '''
    Adapts the weights according to the derivatives.

    Returns the adapted weights.
    '''
    self._c = self._c - np.sum(train_rate * self.d_c, axis=1).reshape(*self._c.shape)
    self._b = self._b - np.sum(train_rate * self.d_b, axis=1).reshape(*self._b.shape)

  @property
  def c(self) -> np.ndarray:
    return self._c.reshape(self.num_neurons, -1)

  @property
  def b(self) -> np.ndarray:
    return self._b.flatten()
