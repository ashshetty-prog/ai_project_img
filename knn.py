from load_data import DigitData
import time
from multiprocessing import Pool
from statistics import mode

from load_data import DigitData

start_time = time.time()


class KnnDigits:
    def __init__(self, k=5):
        self.digit_data = DigitData("digitdata")
        self.distance = []
        self.k = k
        feature_function = self.digit_data.basic_feature_extractor_digit
        self.trainingData = list(map(feature_function, self.digit_data.digit_train_imgs))
        self.testData = list(map(feature_function, self.digit_data.digit_test_imgs))

    def predict(self, image):
        img_a = image
        self.distance = list(map(lambda x: (self.digit_data.digit_train_labels[x[0]], img_a.cosine_distance(x[1])),
                                 enumerate(self.trainingData)))
        # sort the list of tuples by distances in increasing order
        sorted_dist = (sorted(self.distance, key=lambda x: x[1]))
        k_neighbors = sorted_dist[:self.k]
        # select k labels
        klabels = [label for (label, _) in k_neighbors]
        # find the mode of the list
        pred = mode(klabels)
        return pred


if __name__ == '__main__':
    knnd = KnnDigits(5)

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    agents = 5
    chunksize = 3
    with Pool(processes=agents) as pool:
        predictions = pool.map(knnd.predict, knnd.testData, chunksize)
    '''
    predictions = list(map(knnd.predict, knnd.testData))
    print('map output', predictions)

    #for image in knnd.testData:
    #    predictions.append(knnd.predict(image))
    '''

    correct, wrong = (0, 0)
    for n, pred in enumerate(predictions):
        if pred == knnd.digit_data.digit_test_labels[n]:
            correct += 1
        else:
            wrong += 1
    print("The predictions are: ", predictions)
    print("The actual labels are:", knnd.digit_data.digit_test_labels)
    print("No. of correct guesses = {}".format(correct))
    print("No. of wrong guesses = {}".format(wrong))
    print("Percentage accuracy: {}".format((correct * 100) / (correct + wrong)))
    print('execution time', time.time() - start_time)
