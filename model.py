import numpy as np
from neurons import Neuron
from layer import Layer
from typing import List
import loss
from time import time

class Model:
    layers : List[Layer]
    
    def __init__(self, layers : List[Layer]):
        self.layers = layers

    def __print_progress_bar(self, iteration:int, total_iterations:int, prefix:str = ''):
        filledLength = int(100.0 * iteration // total_iterations)
        output = prefix + f"| {int(filledLength)}% |" + '█' * filledLength + ' ' * (100 - filledLength) + '|'
        print(output, end='\r')
        if(iteration == total_iterations):
            print()

    def train(self, train_input:np.ndarray, train_output:np.ndarray, epochs:int=1, loss_function=loss.quadratic_error, train_rate:float=0.1):
        self.errors = []
        self.losses = []
        print(f"Training model for {epochs} epochs over {train_input.shape[0] * train_input.shape[1]} training examples.")
        start = time()
        avg_epoch_duration = 0.0
        for j in range(epochs):
            epoch_errors = [0.0]
            epoch_loss = []
            avg_epoch_duration = -time()
            for i, batch in enumerate(train_input):
                self.__print_progress_bar(i + train_input.shape[0] * j, train_input.shape[0] *epochs, 
                                          f"Epoch {j + 1}/{epochs} | Batch {i + 1}/{train_input.shape[0]} | Average Error: {epoch_errors[-1]:.2f} ")
                current_input = batch.reshape(1, *batch.shape)
                for l in self.layers:
                    current_input = l.propagate(current_input)
                epoch_errors.append(np.average(np.abs(train_output[i] - current_input[0,:])))
                loss_i, deriv = loss_function(current_input[:,0], train_output[i])
                epoch_loss.append(np.average(loss_i))
                deriv = deriv.reshape(1, batch.shape[0], -1)

                for l in reversed(self.layers):
                    deriv = l.back_propagate(deriv, train_rate)
            self.errors.append(np.average(np.array(epoch_errors[1:])))
            self.losses.append(np.average(np.array(epoch_loss)))
            avg_epoch_duration += time()
        self.__print_progress_bar(epochs, epochs, 
                                  f"Epoch {epochs}/{epochs} | Batch {train_input.shape[0]}/{train_input.shape[0]} | Average Error: {np.average(np.array(self.errors)):.2f} ")
        print(f"Finished training in {time() - start} seconds.")

    def predict(self, input:np.ndarray) -> np.ndarray:
        print(f"Predicting {input.shape[0] * input.shape[1]} inputs.")
        outputs = []
        for i, batch in enumerate(input):
            current_input = batch.reshape(1, *batch.shape) 
            self.__print_progress_bar(i, input.shape[0], f"Batch {i + 1}/{input.shape[0]} ")
            for l in self.layers:
                current_input = l.propagate(current_input)
            outputs.append(current_input)
        self.__print_progress_bar(input.shape[0], input.shape[0], f"Batch {input.shape[0]}/{input.shape[0]} ")
        return np.array(outputs).reshape(input.shape[0], input.shape[1], -1)

    def validate(self, input:np.ndarray, expected_output:np.ndarray) -> float:
        output = self.predict(input)
        print("Validtaing outputs.")
        error = np.abs(output - expected_output)
        return np.average(error)