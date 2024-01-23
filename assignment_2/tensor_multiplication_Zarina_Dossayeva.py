import torch

def create_tensor_of_val(dimensions, val):
    """
    Create a tensor of the given dimensions, filled with the value of `val`.
    """
    res = torch.full(dimensions, val)
    return res

def calculate_elementwise_product(A, B):
    """
    Calculate the elementwise product of the two tensors A and B.
    """
    res = A * B
    return res

def calculate_matrix_product(X, W):
    """
    Calculate the product of the two tensors X and W. (sum {x_i * w_i})
    """
    res = torch.matmul(X, W.T)
    return res

def calculate_matrix_prod_with_bias(X, W, b):
    """
    Calculate the product of the two tensors X and W. (sum {x_i * w_i}) and add the bias.
    """
    res = torch.matmul(X, W.T) + b
    return res

def calculate_activation(sum_total):
    """
    Calculate a step function as an activation of the neuron.
    """
    res = torch.heaviside(sum_total, torch.tensor(0.0))
    return res 

def calculate_output(X, W, b):
    """
    Calculate the output of the neuron.
    """
    sum_total = calculate_matrix_prod_with_bias(X, W, b)
    res = calculate_activation(sum_total)
    return res