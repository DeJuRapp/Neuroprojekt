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
  old_in:np.ndarray
  kernel_dimension:List[int]
  horizontal_stride:int
  vertical_stride:int
  num_horizontal:int
  num_vertical:int
  neuron_type:Neuron

  def __init__(self, input_dimension:np.ndarray, stride_horizontal:int, stride_vertical:int, kernel_dimensions:List[int], neuron_type:Neuron):
    if (input_dimension[2] % kernel_dimensions[2]) != 0:
      raise ValueError("Color channel cannot be mapped by this kernel shape.")

    self.num_horizontal = int((input_dimension[0] - int(kernel_dimensions[0] / 2) - 1) / stride_horizontal)
    self.num_vertical = int((input_dimension[1] - int(kernel_dimensions[0] / 2) - 1) / stride_vertical)

    self.kernel_dimension = kernel_dimensions
    self.vertical_stride = stride_vertical
    self.horizontal_stride = stride_horizontal

    self.neuron_type = neuron_type

    self.neuron_data = neuron_type.init_zeros(self.num_horizontal * self.num_vertical, kernel_dimensions)
    self.old_in = np.empty((self.neuron_data.shape[0], *self.kernel_dimension))
  
  def __create_sub_images(self, image:np.ndarray) -> np.ndarray:
    neuron_index = 0
    for i in range(0, image.shape[0] - int(self.kernel_dimension[0] / 2) - 1, self.horizontal_stride):
      for j in range(0, image.shape[1] - int(self.kernel_dimension[1] / 2) - 1, self.vertical_stride):
        self.old_in[neuron_index] = image[i:i+self.kernel_dimension[0], j:j+self.kernel_dimension[1], 0:3]
        neuron_index += 1
    return self.old_in

  def propagate(self, image:np.ndarray) -> np.ndarray:
    inputs = self.__create_sub_images(image)
    self.old_out = self.neuron_type.propagate(inputs, self.neuron_data)
    return self.old_out

  def back_propagate(self, derivative:np.ndarray) -> np.ndarray:
    d_w, d_x = self.neuron_type.back_propagate(self.old_in, self.neuron_data, self.old_out, derivative)