from typing import Tuple, List
import numpy as np


STANDARD_DEVIATION = 1


class Neuron():
  @staticmethod
  def init_zeros(neurons:int, weight_shape: np.ndarray) -> np.ndarray:
    '''
    Returns:
      A numpy array containing zeros such that this array can be passed into propagate.
    '''

  @staticmethod
  def init_random(neurons:int, weight_shape: np.ndarray) -> np.ndarray:
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''

  @staticmethod
  def propagate(x:np.ndarray, w:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of neurons defined by the array w.
    '''

  @staticmethod
  def back_propagate(x:np.ndarray, w:np.ndarray, y:np.ndarray, derivatives:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns the derivatives for w and x.
    '''

  @staticmethod
  def adapt(w:np.ndarray, d_w:np.ndarray, train_rate:float) -> np.ndarray:
    '''
    Adapts the weights according to the derivatives.

    Returns the adapted weights.
    '''

class RBF(Neuron):
  @staticmethod
  def init_zeros(neurons:int, kernel_dims: Tuple[int]) -> np.ndarray:
    '''
    Returns:
      A (n, kernel_dims) numpy array containing zeros.
    '''
    return np.zeros((neurons, *kernel_dims))

  @staticmethod
  def init_random(neurons:int, weight_shape: np.ndarray, min:float, max:float) -> np.ndarray:
    '''
    Returns:
      A numpy array containing random weights such that this array can be passed into propagate.
    '''
    return np.random.uniform(min, max, (neurons, *weight_shape))

  @staticmethod
  def propagate(x:np.ndarray, c:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of RBF neurons defined by the array w.

    Args:
      x: A (n, w, h, ch) numpy array representing the input image
      c: A (n, w, h, ch) numpy array where n is the number of neurons
    Returns:
      y: A (n,) numpy array for the outputs.
    '''
    return np.exp(-np.sum(np.square(x.reshape(x.shape[0], -1) - c.reshape(c.shape[0], -1)), axis=1) / (2 * (STANDARD_DEVIATION ** 2))).reshape(1, c.shape[0])
  
  @staticmethod
  def adapt(w:np.ndarray, d_w:np.ndarray, train_rate:float) -> np.ndarray:
    return w - train_rate * d_w

  @staticmethod
  def back_propagate(x:np.ndarray, c:np.ndarray, y:np.ndarray, derivatives:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns the derivatives for c and x.

    Args:
      x: A (n, w, h, ch) numpy array representing the input image
      c: A (n, w, h, ch) numpy array where n is the number of neurons
      y: The returned value of propagate
      derivatives: The derivatives for y of the previous layer.
    Returns:
      dy_dc: Derivative of y for c.
      dy_dx: Derivative of y for x.
    '''
    deriv = ((x.reshape(x.shape[0], -1) - c.reshape(c.shape[0], -1)) / (STANDARD_DEVIATION ** 2))
    deriv = deriv * y.reshape(c.shape[0], -1) * derivatives.reshape(c.shape[0], -1)
    return (deriv), (-deriv)