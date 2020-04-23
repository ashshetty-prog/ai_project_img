import p3_utils
from load_data import DigitData, DIGIT_DATUM_WIDTH, DIGIT_DATUM_HEIGHT
import utils

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = utils.Counter()  # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        for label in self.legalLabels:
            for x in range(DIGIT_DATUM_WIDTH):
                for y in range(DIGIT_DATUM_HEIGHT):
                    self.weights[label][(x,y)] = weights[label]


    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        #self.features = trainingData[0].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        errors = []
        for iteration in range(self.max_iterations):
            print('Starting iteration ',iteration)
            err = 0
            for i in range(len(trainingData)):
                vectors = utils.Counter()
                for l in self.legalLabels:
                    vectors[l] = self.weights[l] * trainingData[i]
                    #print('vector', vectors[l])
                guess = vectors.argMax()

                actual = trainingLabels[i]
                if guess != actual:
                    self.weights[guess] = self.weights[guess] - trainingData[i]
                    self.weights[actual] = self.weights[actual] + trainingData[i]
                    err += 1
            errors.append(err)
        #utils.raiseNotDefined()
        return errors

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = utils.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        utils.raiseNotDefined()

        return featuresWeights


def cool_visualization(digit_data):
    """
    I added this just to make sure that the loaded data is correct. But I ended up finding that if you stand far away
    from the monitor, you can actually see the digit. VERY COOL :P

    BTW, I do not recommend inverting the image. I'm just doing it for the visualization. Just get the datums
    directly from DigitData.digit_train_imgs and work with them. I'd recommend checking out the Datum class as well.
    :return:
    """
    for i, datum in enumerate(digit_data.digit_test_imgs):
        inverted_datum = p3_utils.array_invert(datum.get_pixels())
        for row in inverted_datum:
            print(row)
        if i > 4:
            break


if __name__ == '__main__':
    pass