from typing import List
import numpy as np
import matplotlib.pyplot as plt
from neurons import Neuron

class Layer:

  neuron_data:np.ndarray
  old_out:np.ndarray
  __kernel_dimension:List[int]
  __horizontal_stride:int
  __vertical_stride:int
  __num_horizontal:int
  __num_vertical:int

  def __init__(self, input_dimension:np.ndarray, stride_horizontal:int, stride_vertical:int, kernel_dimensions:List[int], neuron_type:Neuron):
    if (input_dimension[2] % kernel_dimensions[2]) != 0:
      raise ValueError("Color channel cannot be mapped by this kernel shape.")

    self.__num_horizontal = int((input_dimension[0] - int(kernel_dimensions[0] / 2)) / stride_horizontal)
    self.__num_vertical = int((input_dimension[1] - int(kernel_dimensions[1] / 2)) / stride_vertical)

    self.__kernel_dimension = kernel_dimensions
    self.__vertical_stride = stride_vertical
    self.__horizontal_stride = stride_horizontal

    self.neuron_data = neuron_type.init_zeros(self.__num_horizontal * self.__num_vertical, kernel_dimensions)
  
  def __create_sub_images(self, image:np.ndarray) -> np.ndarray:
    sub_images = np.empty((self.neuron_data.shape[0], *self.__kernel_dimension))
    neuron_index = 0
    for i in range(int(self.__kernel_dimension[0] / 2), image.shape[0], self.__horizontal_stride):
      for j in range(int(self.__kernel_dimension[1] / 2), image.shape[1], self.__vertical_stride):
        sub_images[neuron_index] = image[i:i+self.__kernel_dimension[0], j:j+self.__kernel_dimension[1], 0:3]
        neuron_index += 1
    return sub_images

  def propagate(self, image:np.ndarray) -> np.ndarray:
    inputs = self.__create_sub_images(image)
    for sub_im in inputs:
      plt.figure()
      plt.imshow(sub_im)

  def back_propagate(self, image:np.ndarray, derivative:np.ndarray) -> np.ndarray:
    pass