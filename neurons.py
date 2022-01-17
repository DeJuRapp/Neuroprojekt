from typing import Tuple, List
import numpy as np

class Neuron():
  '''
  The axis order for neurons should always be (neuron, batch, *weights)
  '''

  num_neurons:int

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

  def back_propagate(self, derivatives:np.ndarray, train_rate:float=0.1) -> np.ndarray:
    '''
    Returns the derivatives for x.
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

  def subsample_EM_init(self, sample_data:np.ndarray, sample_labels:np.ndarray) -> np.ndarray:
    '''
    Initializes the weights of the neurons based on the samples.
    For each sample, one neuron is created.

    The function assumes a one-hot-encoding for the sample labels.

    Args:
      sample_data: A (n,m) numpy array, where n is the number of samples, and m the dimension
                   of the sample vector.
      sample_labels: A (n,k) numpy array, where n is the number of samples and k the dimension
                     for the label encoding.
    Returns:
      A (n, n) numpy array of the feature space for each sample.
    '''
    self._c = sample_data.copy().reshape(sample_data.shape[0], 1, sample_data.shape[1])
    #Preselect number of examples per label
    #Calculate the variance per label.
    for i in range(sample_labels.shape[1]):
      indices = np.where(sample_labels[:,i] == 1.0)[0]
      for index in indices:
        self._sigma[index,0] = np.mean(np.square(sample_data[indices] - sample_data[index]),axis=0)
        self._sigma[index,0] = np.where(self._sigma[index,0] == 0.0, 1.0e-8, self._sigma[index,0])
    features = self.propagate(sample_data.reshape(1, *sample_data.shape))
    self.old_x = self.old_y = None
    return features.reshape(-1, sample_data.shape[0])

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
