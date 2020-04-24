def array_invert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    result = [[] for _ in range(len(array[0]))]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result
