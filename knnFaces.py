import numpy as np
import operator
import utils
from load_data import DigitData, Datum, FaceData
import statistics
from statistics import mode

class KnnFaces:
    def __init__(self, k=5):
        self.face_data = FaceData("facedata")
        self.distance = []
        self.k = k
        featureFunction = self.face_data.basic_feature2_extractor
        self.trainingData = list(map(featureFunction, self.face_data.face_train_images))
        self.testData = list(map(featureFunction, self.face_data.face_test_imgs))


    def predict(self, image):
        img_a = utils.Counter()
        img_b = utils.Counter()
        img_a = image
        for i, train_image in enumerate(self.trainingData):
            img_b = train_image
            self.distance.append((self.face_data.face_train_labels[i], img_a.cosine_distance(img_b)))
            #manhattan_distance(img_b)/(self.face_data.FACE_DATUM_WIDTH*self.face_data.FACE_DATUM_HEIGHT)
        # sort the list of tuples by distances in increasing order
        sorted_dist = (sorted(self.distance, key=lambda x: x[1]))
        k_neighbors = sorted_dist[:self.k]
        print('k neighbors', k_neighbors)
        # select k labels
        klabels = [label for (label, _) in k_neighbors]
        # find the mode of the list
        pred = mode(klabels)
        print('prediction', pred)
        return pred


if __name__ == '__main__':

    knnf = KnnFaces(5)
    # test_features = knnd.get_features_for_data(knnd.digit_data.digit_test_imgs)
    predictions = []

    # cool_visualization(digit_data)

    for image in knnf.testData:
        predictions.append(knnf.predict(image))

    correct, wrong = (0, 0)
    for k, pred in enumerate(predictions):
        if pred == knnf.face_data.face_test_labels[k]:
            correct += 1
        else:
            wrong += 1
    print("The predictions are: ", predictions)
    print("The actual labels are:", knnf.face_data.face_test_labels)
    print("No. of correct guesses = {}".format(correct))
    print("No. of wrong guesses = {}".format(wrong))
    print("Percentage accuracy: {}".format((correct * 100) / (correct + wrong)))