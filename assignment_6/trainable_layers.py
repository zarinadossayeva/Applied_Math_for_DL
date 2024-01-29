from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def define_dense_model_with_hidden_layers(input_length, 
                                          activation_func_array=['sigmoid','sigmoid'],
                                          hidden_layers_sizes=[50, 20],
                                          output_function='softmax',
                                          output_length=10):
                                         
    """Define a dense model with a hidden layer with the following parameters:
    input_length: the number of inputs for the first layer
    activation_func_array: the activation function for the hidden layers
    hidden_layer_sizes: the number of neurons in each of the hidden layer
    output_function: the activation function for the output layer
    output_length: the number of outputs (number of neurons in the output layer)"""

    model = keras.Sequential()
    # Create the input layer
    model.add(layers.Dense(units=hidden_layers_sizes[0], input_shape=(input_length,), activation=activation_func_array[0]))
    # Create the hidden layers
    model.add(layers.Dense(units=hidden_layers_sizes[1], activation=activation_func_array[1]))
    # Create the output layer
    model.add(layers.Dense(units=output_length, activation=output_function))
    return model


def set_layers_to_trainable(model, trainable_layer_numbers):
    """Set the layers of the model to trainable or not trainable.
    model: the model
    trainable_layer_numbers: the numbers of the layers to be set to trainable. set the other layers to not trainable
    """
    # Set the layers to trainable or not trainable
    for i, layer in enumerate(model.layers):
        if i in trainable_layer_numbers:
            layer.trainable = True
        else:
            layer.trainable = False

    return model