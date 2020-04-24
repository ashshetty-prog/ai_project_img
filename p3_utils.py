import numpy as np


def array_invert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    ar = np.array(array)
    flipped_ar = np.flip(ar, axis=1)
    return flipped_ar.tolist()
