import time

import numpy as np

import p3_utils
from load_data import FaceData
from perceptron import PerceptronClassifier

if __name__ == '__main__':
    iterations = 3
    legalLabels = range(10)
    face_data = FaceData("facedata")

    classifier = PerceptronClassifier(legalLabels, iterations)
    featureFunction = face_data.basic_feature_extractor

    trainingDataList = list(map(featureFunction, face_data.face_train_imgs))
    validationDataList = list(map(featureFunction, face_data.face_validation_imgs))
    testDataList = list(map(featureFunction, face_data.face_test_imgs))

    n_train = len(trainingDataList)
    classifier.set_weights(range(10), FaceData.FACE_DATUM_WIDTH, FaceData.FACE_DATUM_HEIGHT)
    # Conduct training and testing

    percentages, runtimes = ([], [])
    for n, n_samples in enumerate(range(n_train // 10, n_train + 1, n_train // 10)):
        start_time = time.time()
        print('Training with {}% of data'.format((n + 1) * 10))
        idx = np.random.choice(n_train, n_samples, replace=False)
        train_img_sample = np.array(trainingDataList)[idx].tolist()
        train_labels_sample = np.array(face_data.face_train_labels)[idx].tolist()
        errors = classifier.train(train_img_sample, train_labels_sample)
        print('errors over 3 iterations', errors)
        print('Validating...')
        validation_guesses = classifier.classify(validationDataList)
        correct = [validation_guesses[i] == face_data.face_validation_labels[i] for i in
                   range(len(face_data.face_validation_labels))].count(True)
        print(str(correct), 'correct out of ', str(len(face_data.face_validation_labels)))
        print('Testing...')
        test_guesses = classifier.classify(testDataList)
        correct = [test_guesses[i] == face_data.face_test_labels[i] for i in
                   range(len(face_data.face_test_labels))].count(True)
        percentage = (100.0 * correct / len(face_data.face_test_labels))
        print(str(correct), 'correct out of ', str(len(face_data.face_test_labels)), 'percentage ', percentage)
        percentages.append(percentage)
        runtime = time.time() - start_time
        print("Runtime taken: ", runtime)
        runtimes.append(runtime)
    p3_utils.plot_line_graph(range(10, 101, 10), percentages, "Accuracy of perceptron for face data",
                             "Percentage of training data used", "Percentage accuracy obtained on test data")
    p3_utils.plot_line_graph(range(10, 101, 10), runtimes, "Runtime of perceptron for face data",
                             "Percentage of training data used", "Runtime taken for trianing and testing")
