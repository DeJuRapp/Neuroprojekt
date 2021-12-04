import numpy as np
from neurons import Neuron
from layer import Layer
from typing import List
import loss

class Model:
    layers : List[Layer]
    
    def __init__(self, layers : List[Layer]):
        self.layers = layers

    def __print_progress_bar(self, iteration:int, total_iterations:int, prefix:str = ''):
        total_iterations -= 1
        filledLength = int(100.0 * iteration // total_iterations)
        print(prefix + f"| {int(filledLength)}% |" + '█' * filledLength + ' ' * (100 - filledLength) + '|', end='\r')
        if(iteration == total_iterations):
            print()

    def train(self, train_input:np.ndarray, train_output:np.ndarray, epochs:int=1, loss_function=loss.quadratic_error, train_rate:float=0.1):
        self.errors = []
        self.losses = []
        print(f"Training model for {epochs} epochs over {train_input.shape[0]} training examples.")
        for j in range(epochs):
            order = np.arange(train_input.shape[0])
            np.random.shuffle(order)
            
            for i, example in enumerate(train_input[order]):
                self.__print_progress_bar(j * train_input.shape[0] + i, train_input.shape[0] * epochs, f"Epoch {j + 1}/{epochs}")
                current_input = example
                for l in self.layers:
                    current_input = l.propagate(current_input)
                self.errors.append(np.abs(train_output[order[i]] - current_input[0]))
                loss_i, deriv = loss_function(current_input[0], train_output[order[i]])
                self.losses.append(loss_i)

                for l in reversed(self.layers):
                    deriv = l.back_propagate(deriv, train_rate)

    def predict(self, input:np.ndarray) -> np.ndarray:
        ret_vals = []
        print(f"Predicting {input.shape[0]} inputs.")
        for i, example in enumerate(input):
            self.__print_progress_bar(i, input.shape[0], f"Input {i + 1}/{input.shape[0]}")
            current_input = example
            for l in self.layers:
                current_input = l.propagate(current_input)
            ret_vals.append(current_input)
        return np.array(ret_vals)