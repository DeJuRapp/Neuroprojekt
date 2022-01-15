from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from neurons import Neuron

class Layer:
  def propagate(self, inputs:np.ndarray) -> np.ndarray:
    pass

  def back_propagate(self, derivative:np.ndarray, train_rate:float) -> np.ndarray:
    pass

class DenseLayer(Layer):
  neurons:Neuron

  def __init__(self, input_dimension:np.ndarray, number_of_neurons:int, neuron_type:Neuron):
    self.neurons = neuron_type
    self.neurons.init_random(number_of_neurons, input_dimension, min=0.0, max=1.0)

  def subsample_EM_init(self, sample_data:np.ndarray, sample_labels:np.ndarray) -> np.ndarray:
    return self.neurons.subsample_EM_init(sample_data, sample_labels)

  def propagate(self, inputs:np.ndarray) -> np.ndarray:
    #Make sure that the input have the correct shape for the neurons.
    return self.neurons.propagate(inputs.reshape(*inputs.shape[:2], -1))

  def back_propagate(self, derivative:np.ndarray, train_rate:float) -> np.ndarray:
    d_x = self.neurons.back_propagate(derivative.reshape(*derivative.shape[:2], -1), train_rate)
    return np.sum(d_x, axis=0).reshape(1, d_x.shape[1], -1)

class BatchNorm(Layer):

  gamma:np.ndarray
  beta:np.ndarray
  epsilon:float
  mu:np.ndarray
  sigma_sqrd:np.ndarray
  rescaled_x:np.ndarray

  def __init__(self, epsilon:float, input_dimension:int):
    self.epsilon = epsilon
    self.gamma = np.random.uniform(0.0, 1.0, (1, 1, input_dimension))
    self.beta = np.random.uniform(0.0, 1.0, (1, 1, input_dimension))

  def propagate(self, inputs:np.ndarray) -> np.ndarray:
    #Calculate mean and variance
    self.mu = np.average(inputs, axis=1)
    x_minus_mu = inputs - self.mu
    self.sigma_sqrd = np.average(np.square(x_minus_mu), axis=1)
    #Rescale inputs
    self.rescaled_x = x_minus_mu / np.sqrt(self.sigma_sqrd + self.epsilon)
    return self.gamma * self.rescaled_x + self.beta

  def back_propagate(self, derivative: np.ndarray, train_rate: float) -> np.ndarray:
    pass
      


class ConvolutionalLayer(Layer):

  neurons:Neuron
  kernel_dimension:int
  stride:int
  num_horizontal:int
  num_vertical:int
  original_shape:Tuple[int]

  def __init__(self, input_dimension:Tuple[int], stride:int, kernel_dimensions:int, neuron_type:Neuron):

    self.original_shape = (1, *input_dimension[1:])

    self.num_horizontal = int(np.ceil((input_dimension[2] - kernel_dimensions) / stride))
    self.num_vertical = int(np.ceil((input_dimension[3] - kernel_dimensions) / stride))

    self.kernel_dimension = kernel_dimensions
    self.stride = stride

    self.neurons = neuron_type
    neuron_type.init_random(self.num_horizontal * self.num_vertical, kernel_dimensions**2, 0.0, 1.0)
  
  def __create_sub_images(self, images:np.ndarray) -> np.ndarray:
    '''
    Args:
      image: Should be a batch of images with the shape (1, batch, width, height)
    '''
    #Shape (neurons, batch, kernel_x1, kernel_x2)
    sub_images = np.empty((self.num_horizontal * self.num_vertical, images.shape[1], self.kernel_dimension, self.kernel_dimension))
    neuron_index = 0
    for i in range(0, images.shape[2] - self.kernel_dimension, self.stride):
      for j in range(0, images.shape[3] - self.kernel_dimension, self.stride):
        sub_images[neuron_index] = images[0, :, i:i+self.kernel_dimension, j:j+self.kernel_dimension]
        neuron_index += 1
    return sub_images.reshape(neuron_index, images.shape[1], -1)

  def __re_assemble_derivatives(self, derivatives:np.ndarray) -> np.ndarray:
    '''
    Inverts the __create_sub_images function for the derivatives.

    Args:
      derivatives: Should be the value returned by neurons.back_propagate
    '''
    derivatives = derivatives.reshape(*derivatives.shape[0:2], self.kernel_dimension, self.kernel_dimension)
    image_derivative = np.zeros(self.original_shape)
    neuron_index = 0
    for i in range(self.num_horizontal):
      for j in range(self.num_vertical):
        image_derivative[0,:,
          i*self.stride:i*self.stride+self.kernel_dimension,
          j*self.stride:j*self.stride+self.kernel_dimension] += derivatives[neuron_index]
        neuron_index += 1
    return image_derivative

  def propagate(self, image:np.ndarray) -> np.ndarray:
    prediction = self.neurons.propagate(self.__create_sub_images(image))
    return prediction.reshape(*prediction.shape[:2], self.num_horizontal, self.num_vertical)

  def back_propagate(self, derivative:np.ndarray, train_rate:float) -> np.ndarray:
    d_x = self.neurons.back_propagate(derivative.reshape(*derivative.shape[:2], -1), train_rate)
    return self.__re_assemble_derivatives(d_x)