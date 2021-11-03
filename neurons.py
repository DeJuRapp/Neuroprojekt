from typing import Tuple
import numpy as np


STANDARD_DEVIATION = 1


class Neuron():
  @staticmethod
  def init_zeros(neurons:int, kernel_dims: np.ndarray) -> np.ndarray:
    '''
    Returns:
      A (n, kernel_dims) numpy array containing zeros.
    '''

  @staticmethod
  def propagate(x:np.ndarray, c:np.ndarray) -> np.ndarray:
    '''
    Propagates the input through a set of RBF neurons defined by the array w.
    '''

  @staticmethod
  def back_propagate(x:np.ndarray, c:np.ndarray, y:np.ndarray, derivatives:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Returns the derivatives for c and x.
    '''

class RBF(Neuron):
  @staticmethod
  def init_zeros(neurons:int, kernel_dims: np.ndarray) -> np.ndarray:
    '''
    Returns:
      A (n, kernel_dims) numpy array containing zeros.
    '''
    return np.zeros((neurons, *kernel_dims))

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
    num_neurons = x.shape[0]
    return np.exp(-(np.square(x.reshape(num_neurons, -1) - c.reshape(num_neurons, -1))) / (2 * (STANDARD_DEVIATION ** 2)))

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
    original_shape = x.shape
    num_neurons = x.shape[0]

    x = x.reshape(num_neurons, -1)
    c = c.reshape(num_neurons, -1)

    factor = (x - c) / (STANDARD_DEVIATION ** 2)

    return (-factor * y).reshape(original_shape), (factor * y).reshape(original_shape)