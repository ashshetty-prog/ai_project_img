# samples.py
# ----------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html


import os
import zipfile
import utils
import digit_perceptron
import numpy as np

import p3_utils

DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
# Module Classes


class DigitData:
    """
    All the image variables are lists of Datums.
    All the labels are just lists of final labels
    """
    def __init__(self, digit_data_path):
        self.digit_path = digit_data_path
        self.digit_train_imgs = load_all_data_in_file(os.path.join(digit_data_path, "trainingimages"), 28, 28)
        self.digit_validation_imgs = load_all_data_in_file(os.path.join(digit_data_path, "validationimages"), 28, 28)
        self.digit_test_imgs = load_all_data_in_file(os.path.join(digit_data_path, "testimages"), 28, 28)
        self.digit_train_labels = load_all_labels_in_file(os.path.join(digit_data_path, "traininglabels"))
        self.digit_validation_labels = load_all_labels_in_file(os.path.join(digit_data_path, "validationlabels"))
        self.digit_test_labels = load_all_labels_in_file(os.path.join(digit_data_path, "testlabels"))

    def basicFeatureExtractorDigit(self, datum):
        """
        Returns a set of pixel features indicating whether
        each pixel in the provided datum is white (0) or gray/black (1)
        """
        a = datum.get_pixels()

        features = utils.Counter()
        for x in range(DIGIT_DATUM_WIDTH):
            for y in range(DIGIT_DATUM_HEIGHT):
                if datum.get_pixel(x, y) > 0:
                    features[(x, y)] = 1
                else:
                    features[(x, y)] = 0
        return features

class Datum:
    """
    A datum is a pixel-level encoding of digits or face/non-face edge maps.

    Digits are from the MNIST dataset and face images are from the
    easy-faces and background categories of the Caltech 101 dataset.


    Each digit is 28x28 pixels, and each face/non-face image is 60x74
    pixels, each pixel can take the following values:
      0: no edge (blank)
      1: gray pixel (+) [used for digits only]
      2: edge [for face] or black pixel [for digit] (#)

    Pixel data is stored in the 2-dimensional array pixels, which
    maps to pixels on a plane according to standard euclidean axes
    with the first dimension denoting the horizontal and the second
    the vertical coordinate:

      28 # # # #      #  #
      27 # # # #      #  #
       .
       .
       .
       3 # # + #      #  #
       2 # # # #      #  #
       1 # # # #      #  #
       0 # # # #      #  #
         0 1 2 3 ... 27 28

    For example, the + in the above diagram is stored in pixels[2][3], or
    more generally pixels[column][row].

    The contents of the representation can be accessed directly
    via the getPixel and getPixels methods.
    """

    def __init__(self, data, width, height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """
        self.height = height
        self.width = width
        if data is None:
            data = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.pixels = p3_utils.array_invert(convert_to_integer(data))

    def get_pixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def get_pixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels

    def get_ascii_string(self):
        """
        Renders the data item as an ascii image.
        """
        rows = []
        data = p3_utils.array_invert(self.pixels)
        for row in data:
            ascii_val = map(ascii_grayscale_conversion_function, row)
            rows.append("".join(ascii_val))
        return "\n".join(rows)

    def __str__(self):
        return self.get_ascii_string()


# Data processing, cleanup and display functions

def load_data_file(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    (Return less then n items if the end of file is encountered).
    """
    datum_width = width
    datum_height = height
    fin = read_lines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        if len(data[0]) < datum_width - 1:
            # we encountered end of file...
            print("Truncating at {} examples (maximum)".format(i))
            break
        items.append(Datum(data, datum_width, datum_height))
    return items


def load_all_data_in_file(filename, width, height):
    """
    Reads all data images from a file and returns a list of Datum objects.
    """
    datum_width = width
    datum_height = height
    fin = read_lines(filename)
    fin.reverse()
    items = []
    for i in range(int(len(fin)/height)):
        data = []
        for j in range(height):
            data.append(list(fin.pop()))
        items.append(Datum(data, datum_width, datum_height))
    return items


def read_lines(filename):
    """Opens a file or reads it from the zip archive data.zip"""
    if os.path.exists(filename):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')


def load_labels_file(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = read_lines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels


def load_all_labels_in_file(filename):
    fin = read_lines(filename)
    labels = []
    for line in fin:
        if line == '':
            break
        labels.append(int(line))
    return labels


def ascii_grayscale_conversion_function(value):
    """
    Helper function for display purposes.
    """
    if value == 0:
        return ' '
    elif value == 1:
        return '+'
    elif value == 2:
        return '#'


def integer_conversion_function(character):
    """
    Helper function for file reading.
    """
    if character == ' ':
        return 0
    elif character == '+':
        return 1
    elif character == '#':
        return 2


def convert_to_integer(data):
    """
    Helper function for file reading.
    """
    for row in data:
        for i in range(len(row)):
            row[i] = integer_conversion_function(row[i])
    return data


# Testing

def _test():
    import doctest
    doctest.testmod()  # Test the interactive sessions in function comments
    n = 1
    #  items = loadDataFile("facedata/facedatatrain", n,60,70)
    #  labels = loadLabelsFile("facedata/facedatatrainlabels", n)
    items = load_data_file("digitdata/trainingimages", n, 28, 28)
    labels = load_labels_file("digitdata/traininglabels", n)
    '''
    for i in range(1):
        print(items[i])
        print(items[i])
        print(items[i].height)
        print(items[i].width)
        print(dir(items[i]))
        print(items[i].get_pixels())
        print(labels[i])
    '''

if __name__ == "__main__":
    _test()
    iterations = 3
    input_data = input('enter the type of data d/f')
    classifier_type = input('Enter the type of classifier n/p')
    if input_data == 'd':
        legalLabels = range(10)
        digit_data = DigitData("digitdata")
    else:
        pass
        #create face object

    if classifier_type == 'p':
        classifier = digit_perceptron.PerceptronClassifier(legalLabels, iterations)
        featureFunction = digit_data.basicFeatureExtractorDigit
        # digit_perceptron.cool_visualization(digit_data)

    "Extracting features..."
    trainingData = map(featureFunction, digit_data.digit_train_imgs)
    validationData = map(featureFunction, digit_data.digit_validation_imgs)
    testData = map(featureFunction, digit_data.digit_test_imgs)

    #print('training data',next(trainingData))
    trainingDataList = list(trainingData)
    validationDataList = list(validationData)
    testDataList = list(testData)
    size = len(trainingDataList)
    classifier.setWeights(range(10))
    #for label in legalLabels:
        #print('weights',classifier.weights[label])s
    # Conduct training and testing
    for i in np.arange(0.8, 1, 0.1):
        index = int(size*i)
        #print('index ', index)
        print('Training...')
        errors = classifier.train(trainingDataList[:index], digit_data.digit_train_labels, validationData, digit_data.digit_validation_labels)
        print('errors over 3 iterations', errors)
        print('Validating...')
        guesses = classifier.classify(validationDataList)
        correct = [guesses[i] == digit_data.digit_validation_labels[i] for i in range(len(digit_data.digit_validation_labels))].count(True)
        print(str(correct), 'correct out of ',str(len(digit_data.digit_validation_labels)))
        #" (%.1f%%).") % (100.0 * correct / len(validationLabels)) len(digit_data.digit_validation_labels)
        print('Testing...')
        guesses = classifier.classify(testDataList)
        correct = [guesses[i] == digit_data.digit_test_labels[i] for i in range(len(digit_data.digit_test_labels))].count(True)
        print(str(correct), 'correct out of ',str(len(digit_data.digit_test_labels)),'percentage', (100.0 * correct / len(digit_data.digit_test_labels)))