import p3_utils


class NaiveBayesPredictor:
    def __init__(self, train_images, train_labels, all_labels):
        self.label_prob = {}
        self.feature_prob = {}
        self.all_labels = all_labels
        self.feature_given_label_prob = {}
        self.train_features = p3_utils.get_features_for_data(train_images)
        self.build_probabilities(self.train_features, train_labels)

    def build_probabilities(self, train_features, train_labels):
        feature_given_label_counter = {}
        feature_counter = [1 for _ in range(len(train_features[0]))]
        for label in self.all_labels:
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
        for label in self.all_labels:
            prob = 1
            for i, ele in enumerate(features):
                if ele == 1:
                    prob *= (self.feature_given_label_prob[label][i] / self.feature_prob[i])
                else:
                    prob *= ((1 - self.feature_given_label_prob[label][i]) / (1 - self.feature_prob[i]))
            prob_labels.append(prob * self.label_prob[label])
        return prob_labels.index(max(prob_labels))
