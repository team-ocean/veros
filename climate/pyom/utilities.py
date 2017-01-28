import numpy as np

def pad_z_edges(array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = np.empty(array.shape[0] + 2)
        newarray[1:-1] = array
        newarray[0] = array[0]
        newarray[-1] = array[-1]
    elif array.ndim >= 3:
        a = list(array.shape)
        a[2] += 2
        newarray = np.empty(a)
        newarray[:,:,1:-1,...] = array
        newarray[:,:,0,...] = array[:,:,0,...]
        newarray[:,:,-1,...] = array[:,:,-1,...]
    else:
        raise ValueError("Array to pad needs to have 1 or at least 3 dimensions")
    return newarray
