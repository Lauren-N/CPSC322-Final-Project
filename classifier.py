import numpy as np

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        categorical(list of bool): Indicates which features are categorical
    """
    def __init__(self, n_neighbors=3, categorical=None):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
            categorical(list of bool): List indicating which features are categorical
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.categorical = categorical if categorical is not None else []

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def _calculate_distance(self, sample1, sample2):
        """Calculates the distance between two samples, considering categorical features.

        Args:
            sample1(list): A list of feature values for the first sample
            sample2(list): A list of feature values for the second sample

        Returns:
            float: The distance between the two samples
        """
        distance = 0
        for i in range(len(sample1)):
            if i < len(self.categorical) and self.categorical[i]:
                # If the feature is categorical, use 0 for same values, 1 for different values
                distance += 0 if sample1[i] == sample2[i] else 1
            else:
                # Otherwise, calculate Euclidean distance (for numeric values)
                distance += (sample1[i] - sample2[i]) ** 2
        return np.sqrt(distance)

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for test_sample in X_test:
            dist = []
            for train_sample in self.X_train:
                dist.append(self._calculate_distance(test_sample, train_sample))
            nearest_indices = np.argsort(dist)[:self.n_neighbors]
            distances.append([dist[i] for i in nearest_indices])
            neighbor_indices.append(nearest_indices.tolist())

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, neighbor_indices = self.kneighbors(X_test)

        y_predicted = []

        for i, indices in enumerate(neighbor_indices):
            closest_labels = [self.y_train[i] for i in indices]

            # Count occurrences of each label
            label_counts = {}
            for label in closest_labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            # Find the label with the maximum count
            most_common_label = max(label_counts, key=label_counts.get)
            y_predicted.append(most_common_label)

        return y_predicted


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.posteriors = {}

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # Count occurrences of each label in y_train to compute priors
        label_counts = {}
        for label in y_train:
            label_counts[label] = label_counts.get(label, 0) + 1

        # Calculate priors
        total_samples = len(y_train)
        self.priors = {label: count / total_samples for label, count in label_counts.items()}

        # Initialize posteriors
        self.posteriors = {label: [{} for _ in range(len(X_train[0]))] for label in label_counts}

        # Count occurrences of each attribute value given each label
        for i in range(len(X_train)):
            label = y_train[i]
            for j, value in enumerate(X_train[i]):
                if value not in self.posteriors[label][j]:
                    self.posteriors[label][j][value] = 1
                else:
                    self.posteriors[label][j][value] += 1

        # Calculate posteriors
        for label, feature_counts in self.posteriors.items():
            for j, value_counts in enumerate(feature_counts):
                total_label_count = label_counts[label]
                self.posteriors[label][j] = {
                    value: count / total_label_count for value, count in value_counts.items()
                }

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instance in X_test:
            label_probs = {}
            for label in self.priors:
                # Start with the prior probability of the label
                label_prob = self.priors[label]
                # Multiply by each attribute's conditional probability given the label
                for j, value in enumerate(instance):
                    # Use a small probability if the value hasn't been seen in training data
                    label_prob *= self.posteriors[label][j].get(value, 1e-6)
                label_probs[label] = label_prob

            # Predict the label with the highest probability
            best_label = max(label_probs, key=label_probs.get)
            y_predicted.append(best_label)

        return y_predicted
