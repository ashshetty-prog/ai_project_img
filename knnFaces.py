import time
from multiprocessing import Pool
from statistics import mode

from load_data import FaceData

start_time = time.time()


class KnnFaces:
    def __init__(self, k=5):
        self.face_data = FaceData("facedata")
        self.distance = []
        self.k = k
        feature_function = self.face_data.basic_feature2_extractor
        self.trainingData = list(map(feature_function, self.face_data.face_train_images))
        self.testData = list(map(feature_function, self.face_data.face_test_imgs))

    def predict(self, image):
        img_a = image
        self.distance = list(map(lambda x: (self.face_data.face_train_labels[x[0]], img_a.cosine_distance(x[1])),
                                 enumerate(self.trainingData)))
        sorted_dist = (sorted(self.distance, key=lambda x: x[1]))
        k_neighbors = sorted_dist[:self.k]
        # select k labels
        klabels = [label for (label, _) in k_neighbors]
        # find the mode of the list
        return mode(klabels)


if __name__ == '__main__':

    knnf = KnnFaces(5)
    # test_features = knnd.get_features_for_data(knnd.digit_data.digit_test_imgs)

    # Run this with a pool of 5 agents having a chunksize of 3 until finished
    agents = 5
    chunksize = 3
    with Pool(processes=agents) as pool:
        predictions = pool.map(knnf.predict, knnf.testData, chunksize)
    # cool_visualization(digit_data)

    correct, wrong = (0, 0)
    for n, pred in enumerate(predictions):
        if pred == knnf.face_data.face_test_labels[n]:
            correct += 1
        else:
            wrong += 1
    print("The predictions are: ", predictions)
    print("The actual labels are:", knnf.face_data.face_test_labels)
    print("No. of correct guesses = {}".format(correct))
    print("No. of wrong guesses = {}".format(wrong))
    print("Percentage accuracy: {}".format((correct * 100) / (correct + wrong)))
    print('execution time', time.time() - start_time)
