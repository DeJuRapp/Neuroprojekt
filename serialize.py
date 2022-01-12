import pickle
from model import Model
import layer
import neurons
from pathlib import Path

def serialize_neurons(n:neurons.Neuron) -> dict:
    neuron_data = {}
    if type(n) == neurons.RBF:
        n:neurons.RBF = n
        neuron_data["type"] = "RBF"
        neuron_data["c"] = n._c
        neuron_data["sigma"] = n._sigma
    else:
        print(f"Unkown neuron type: {type(n)}.")
    return neuron_data


def serialize_layer(l:layer.Layer) -> dict:
    layer_data = {}
    if type(l) == layer.DenseLayer:
        l:layer.DenseLayer = l
        layer_data["type"] = "Dense"
        layer_data["neurons"] = serialize_neurons(l.neurons)
    elif type(l) == layer.ConvolutionalLayer:
        l:layer.ConvolutionalLayer = l
        layer_data["type"] = "Conv"
        layer_data["neurons"] = serialize_neurons(l.neurons)
        layer_data["original_shape"] = l.original_shape
        layer_data["num_vertical"] = l.num_vertical
        layer_data["num_horizontal"] = l.num_horizontal
        layer_data["kernel_dimension"] = l.kernel_dimension
        layer_data["stride"] = l.stride
    else:
        print(f"Unkown layer type: {type(l)}.")
    return layer_data

def serialize_model(model:Model, path:Path):
    model_data = []
    for l in model.layers:
        model_data.append(serialize_layer(l))
    with open(path, "wb") as f:
        pickle.dump(model_data, f)