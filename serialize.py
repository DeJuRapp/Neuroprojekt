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

def deserialize_neurons(neuron_data:dict) -> neurons.Neuron:
    n = neurons.Neuron()
    if neuron_data["type"] == "RBF":
        n:neurons.RBF = neurons.RBF()
        n._c = neuron_data["c"]
        n._sigma = neuron_data["sigma"]
        n.num_neurons = n._c.shape[0]
    else:
        print(f"Unkown neuron type: {neuron_data['type']}.")
    return n



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

def deserialize_layer(layer_data:dict) -> layer.Layer:
    l = layer.Layer()
    if layer_data["type"] == "Dense":
        l:layer.DenseLayer = layer.DenseLayer(1, 1, neurons.Neuron())
        l.neurons = deserialize_neurons(layer_data["neurons"])
    elif layer_data["type"] == "Conv":
        l:layer.ConvolutionalLayer = layer.ConvolutionalLayer((1,1,1,1), 1, 1, neurons.RBF())
        l.neurons = deserialize_neurons(layer_data["neurons"])
        l.kernel_dimension = layer_data["kernel_dimension"]
        l.num_horizontal = layer_data["num_horizontal"]
        l.num_vertical = layer_data["num_vertical"]
        l.original_shape = layer_data["original_shape"]
        l.stride = layer_data["stride"]
    else:
        print(f"Unkown layer type: {layer_data['type']}.")
    return l

def serialize_model(model:Model, path:Path):
    model_data = []
    for l in model.layers:
        model_data.append(serialize_layer(l))
    with open(path, "wb") as f:
        pickle.dump(model_data, f)

def deserialize_model(path:Path) -> Model:
    with open(path, "rb") as f:
        model_data = pickle.load(f)
    layers = []
    for l in model_data:
        layers.append(deserialize_layer(l))
    return Model(layers)
