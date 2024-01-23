import numpy as np

def rmse(predictions, targets):
    """
    Computes the Root Mean Square Error (RMSE) between predictions and targets.
    predictions: list of predicted values
    targets: list of actual values
    """
    predictions_array = np.array(predictions)
    targets_array = np.array(targets)
    rmse = np.sqrt(np.mean((predictions_array - targets_array) ** 2))
    return rmse