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
  _W : np.ndarray
  x_c : np.ndarray
  old_y : np.ndarray
  matmul_axes:List[Tuple[int]]

  def init_random(self, neurons:int, weight_shape: int, min:float, max:float):
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''
    self.num_neurons = neurons
    self._c = np.random.uniform(min, max, (neurons, 1, weight_shape))
    self._W = np.zeros((neurons, 1, weight_shape, weight_shape))
    self._W[:,:,np.array([(i,i) for i in range(weight_shape)])] = weight_shape / 3.0
    self.matmul_axes = [(-2,-1), (-2,-1), (-2,-1)]

  @staticmethod
  def __transpose_batch(vector:np.ndarray) -> np.ndarray:
    '''
    Accepts a vector of shape (neurons, batch, weight, 1) and returns (neurons, batch, 1, weight)
    '''
    return vector.transpose((0, 1, 3, 2))

  def propagate(self, x:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of neurons defined by the array w.
    '''
    self.x_c:np.ndarray = x - self._c
    self.x_c = self.x_c.reshape(*self.x_c.shape, 1)
    exponent = np.matmul(self.__transpose_batch(self.x_c), self._W, axes=self.matmul_axes)
    exponent = np.matmul(exponent, self.x_c, axes=self.matmul_axes)
    self.old_y = np.exp(-exponent).reshape(exponent.shape[:-1]).T
    return self.old_y

  def back_propagate(self, derivatives:np.ndarray, train:bool=True, train_rate:float=0.1) -> np.ndarray:
    '''
    Returns the derivatives for x.
    '''
    d_c = np.matmul(2 * self._W, self.x_c, axes=self.matmul_axes)
    d_c = d_c.reshape(d_c.shape[:-1]) * (self.old_y * derivatives).T

    if train:
      d_W = np.matmul(self.x_c, self.__transpose_batch(self.x_c), axes=self.matmul_axes) * (self.old_y * derivatives).reshape(1, *self.old_y.shape).T
      self._W = self._W - np.sum(train_rate * d_W, axis=1).reshape(self._W.shape)
      d_W = None
      
      self._c = self._c - np.sum(train_rate * d_c, axis=1).reshape(self._c.shape)

    #Delete old stored results
    self.old_y = None
    self.x_c = None

    return -d_c

  @property
  def c(self) -> np.ndarray:
    return self._c.reshape(self.num_neurons, -1)
