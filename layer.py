from typing import List
import numpy as np
import matplotlib.pyplot as plt
from neurons import Neuron

class Layer:
  def propagate(self, image:np.ndarray) -> np.ndarray:
    pass

  def back_propagate(self, image:np.ndarray, derivative:np.ndarray) -> np.ndarray:
    pass

class ConvolutionalLayer(Layer):

  neuron_data:np.ndarray
  old_out:np.ndarray
  __kernel_dimension:List[int]
  __horizontal_stride:int
  __vertical_stride:int
  __num_horizontal:int
  __num_vertical:int
  __neuron_type:Neuron

  def __init__(self, input_dimension:np.ndarray, stride_horizontal:int, stride_vertical:int, kernel_dimensions:List[int], neuron_type:Neuron):
    if (input_dimension[2] % kernel_dimensions[2]) != 0:
      raise ValueError("Color channel cannot be mapped by this kernel shape.")

    self.__num_horizontal = int((input_dimension[0] - int(kernel_dimensions[0] / 2) - 1) / stride_horizontal)
    self.__num_vertical = int((input_dimension[1] - int(kernel_dimensions[0] / 2) - 1) / stride_vertical)

    self.__kernel_dimension = kernel_dimensions
    self.__vertical_stride = stride_vertical
    self.__horizontal_stride = stride_horizontal

    self.__neuron_type = neuron_type

    self.neuron_data = neuron_type.init_zeros(self.__num_horizontal * self.__num_vertical, kernel_dimensions)
  
  def __create_sub_images(self, image:np.ndarray) -> np.ndarray:
    sub_images = np.empty((self.neuron_data.shape[0], *self.__kernel_dimension))
    neuron_index = 0
    for i in range(0, image.shape[0] - int(self.__kernel_dimension[0] / 2) - 1, self.__horizontal_stride):
      for j in range(0, image.shape[1] - int(self.__kernel_dimension[1] / 2) - 1, self.__vertical_stride):
        sub_images[neuron_index] = image[i:i+self.__kernel_dimension[0], j:j+self.__kernel_dimension[1], 0:3]
        neuron_index += 1
    return sub_images

  def propagate(self, image:np.ndarray) -> np.ndarray:
    inputs = self.__create_sub_images(image)
    self.old_out = self.__neuron_type.propagate(inputs, self.neuron_data)
    return self.old_out

  def back_propagate(self, image:np.ndarray, derivative:np.ndarray) -> np.ndarray:
    pass