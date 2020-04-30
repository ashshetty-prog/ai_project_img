from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from load_data import Datum


def array_invert(array):
    """
    Inverts a matrix stored as a list of lists.
    """
    result = [[] for _ in range(len(array[0]))]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result


def get_blocks(arr: np.ndarray, nrows, ncols):
    n_vert_blocks = arr.shape[1] // ncols
    n_horiz_blocks = arr.shape[0] // nrows
    blocks = list()
    horiz_split = np.split(arr, n_horiz_blocks)
    for split_arr in horiz_split:
        blocks.extend(np.split(split_arr, n_vert_blocks, axis=1))
    return blocks


def get_features_from_blocks(blocks):
    features = list()
    for block in blocks:
        if list(block.flatten()).count(0) != len(block.flatten()):
            features.append(1)
        else:
            features.append(0)
    return features


def get_features_for_data(data: List[Datum]):
    features = []
    for d in data:
        pixels = np.array(d.get_pixels())
        blocks = get_blocks(pixels, 2, 2)
        features.append(get_features_from_blocks(blocks))

    return features


def plot_line_graph(x, y, plot_title, x_label, y_label):
    seaborn.lineplot(x, y)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def cool_visualization(face_data):
    """
    I added this just to make sure that the loaded data is correct. But I ended up finding that if you stand far away
    from the monitor, you can actually see the digit. VERY COOL :P

    BTW, I do not recommend inverting the image. I'm just doing it for the visualization. Just get the datums
    directly from DigitData.digit_train_imgs and work with them. I'd recommend checking out the Datum class as well.
    :return:
    """
    for i, datum in enumerate(face_data.face_test_imgs):
        inverted_datum = array_invert(datum.get_pixels())
        for row in inverted_datum:
            print(row)
        if i > 4:
            break
