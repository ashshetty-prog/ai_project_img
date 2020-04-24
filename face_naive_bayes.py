from typing import List

import numpy as np

from load_data import FaceData, Datum


class NaiveBayesFace:
    def __init__(self):
        self.face_data = FaceData("facedata")
        self.label_prob = {}
        self.feature_prob = {}
        self.feature_given_label_prob = {}
        self.train_features = self.get_features_for_data(self.face_data.face_train_images)
        self.build_probabilities(self.train_features, self.face_data.face_train_labels)

    def get_blocks(self, arr: np.ndarray, nrows, ncols):
        n_vert_blocks = arr.shape[1] // ncols
        n_horiz_blocks = arr.shape[0] // nrows
        blocks = list()
        horiz_split = np.split(arr, n_horiz_blocks)
        for split_arr in horiz_split:
            blocks.extend(np.split(split_arr, n_vert_blocks, axis=1))
        return blocks

    def get_features_from_blocks(self, blocks):
        features = list()
        for block in blocks:
            if list(block.flatten()).count(0) != len(block.flatten()):
                features.append(1)
            else:
                features.append(0)
        return features

    def get_features_for_data(self, data: List[Datum]):
        features = []
        for d in data:
            pixels = np.array(d.get_pixels())
            blocks = self.get_blocks(pixels, 2, 2)
            features.append(self.get_features_from_blocks(blocks))

        return features

    def build_probabilities(self, train_features, train_labels):
        feature_given_label_counter = {}
        feature_counter = [1 for _ in range(len(train_features[0]))]
        for label in range(2):
            self.label_prob[label] = train_labels.count(label) / len(train_labels)
            feature_given_label_counter[label] = [1 for _ in range(len(train_features[0]))]

        for i, f in enumerate(train_features):
            label = train_labels[i]
            for j, element in enumerate(f):
                feature_given_label_counter[label][j] += element
                feature_counter[j] += element
        for label in feature_given_label_counter:
            self.feature_given_label_prob[label] = [x / train_labels.count(label) for x in
                                                    feature_given_label_counter[label]]
        self.feature_prob = [x / len(train_features) for x in feature_counter]

    def predict(self, features):
        prob_labels = []
        for label in range(2):
            prob = 1
            for i, ele in enumerate(features):
                if ele == 1:
                    prob *= (self.feature_given_label_prob[label][i] / self.feature_prob[i])
                else:
                    prob *= ((1 - self.feature_given_label_prob[label][i]) / (1 - self.feature_prob[i]))
            prob_labels.append(prob * self.label_prob[label])
        return prob_labels.index(max(prob_labels))


if __name__ == '__main__':
    nbf = NaiveBayesFace()
    # for img in nbd.face_data.face_test_imgs:
    #     print(len(img.get_pixels()))
    #     pixel_list = img.get_pixels()
    #     for p in pixel_list:
    #         print(len(p))
    #     break
    test_features = nbf.get_features_for_data(nbf.face_data.face_test_imgs)
    predictions = []
    for feature in test_features:
        predictions.append(nbf.predict(feature))

    correct, wrong = (0, 0)
    for k, pred in enumerate(predictions):
        if pred == nbf.face_data.face_test_labels[k]:
            correct += 1
        else:
            wrong += 1
    print("The predictions are: ", predictions)
    print("The actual labels are:", nbf.face_data.face_test_labels)
    print("No. of correct guesses = {}".format(correct))
    print("No. of wrong guesses = {}".format(wrong))
    print("Percentage accuracy: {}".format((correct * 100) / (correct + wrong)))