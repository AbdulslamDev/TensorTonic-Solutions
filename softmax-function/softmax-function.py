import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    x = np.array(x , dtype=float)

    if x.ndim == 1:
            return (np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))))
    else:
        x = x - np.max(x , axis=1 , keepdims=True)
        return (np.exp(x) / np.sum(np.exp(x), axis=1 , keepdims=True))
        
    # Write code here

    pass