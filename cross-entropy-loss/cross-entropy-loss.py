import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    y_true = np.array(y_true , dtype="int")
    y_pred = np.array(y_pred , dtype="float")
    N = y_true.shape[0]
    
    correct_props = y_pred[np.arange(N), y_true]
    
    return -np.mean(np.log(correct_props))
    pass