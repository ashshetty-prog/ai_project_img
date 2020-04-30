import utils


class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legal_labels, max_iterations):
        self.legal_labels = legal_labels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legal_labels:
            self.weights[label] = utils.Counter()  # this is the data-structure you should use

    def set_weights(self, weights, width, height):
        assert len(weights) == len(self.legal_labels);
        for label in self.legal_labels:
            for x in range(width):
                for y in range(height):
                    self.weights[label][(x, y)] = weights[label]

    def train(self, training_data, training_labels):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        errors = []
        for iteration in range(self.max_iterations):
            print('Starting iteration ', iteration)
            err = 0
            for i in range(len(training_data)):
                vectors = utils.Counter()
                for l in self.legal_labels:
                    vectors[l] = self.weights[l] * training_data[i]
                    # print('vector', vectors[l])
                guess = vectors.argMax()

                actual = training_labels[i]
                if guess != actual:
                    self.weights[guess] = self.weights[guess] - training_data[i]
                    self.weights[actual] = self.weights[actual] + training_data[i]
                    err += 1
            errors.append(err)
        # utils.raiseNotDefined()
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
            for l in self.legal_labels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
