
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