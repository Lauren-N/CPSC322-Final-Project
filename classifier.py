import numpy as np

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.
    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.
    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None

    def calculate_entropy(self, partition):
        """
        Calculate entropy for a given partition.

        Entropy is a measure of the amount of uncertainty or impurity in a dataset.
        This function calculates the entropy of a partition by summing the 
        proportion of each label in the partition multiplied by the log base 2 
        of that proportion, and then taking the negative of that sum.

        Parameters:
        partition (list of lists): A list of instances, where each instance is a 
                                   list and the last element is the label.

        Returns:
        float: The entropy of the partition. If the partition is empty, returns 0.
        """
        total = len(partition)
        if total == 0:
            return 0
        counts = {}
        for instance in partition:
            label = instance[-1]
            counts[label] = counts.get(label, 0) + 1
        return -sum((count / total) * np.log2(count / total) for count in counts.values())


    def select_attribute(self, instances, attributes):
        """
        Selects the best attribute to split the instances based on the minimum entropy.
        Args:
            instances (list of list): The dataset where each inner list represents an instance.
            attributes (list of str): The list of attribute names to consider for splitting.
        Returns:
            str: The attribute with the lowest weighted entropy, indicating the best split.
        """
        min_entropy = float('inf')
        best_attribute = None

        for attribute in attributes:
            partitions = self.partition_instances(instances, attribute)
            weighted_entropy = 0
            total_instances = len(instances)

            for partition in partitions.values():
                weighted_entropy += (len(partition) / total_instances) * self.calculate_entropy(partition)

            if weighted_entropy < min_entropy:
                min_entropy = weighted_entropy
                best_attribute = attribute

        return best_attribute

    def partition_instances(self, instances, attribute):
        """
        Partitions a list of instances based on the values of a specified attribute.
        Args:
            instances (list of list): The dataset to be partitioned, where each inner list represents an instance.
            attribute (str): The attribute on which to partition the instances.
        Returns:
            dict: A dictionary where keys are attribute values and values are lists of instances that have the corresponding attribute value.
        """
        str_attribute = str(attribute)
        att_index = self.header.index(str_attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def all_same_class(self, instances):
        """
        Check if all instances belong to the same class.

        Args:
            instances (list of list): A list of instances, where each instance is a list and the last element is the class label.

        Returns:
            bool: True if all instances have the same class label, False otherwise.
        """
        first_class = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_class:
                return False
        # get here, then all same class labels
        return True

    def tdidt(self, current_instances, available_attributes):
        """
        Perform the Top-Down Induction of Decision Trees (TDIDT) algorithm to build a decision tree.
        Args:
            current_instances (list of list): The current subset of instances to be used for building the tree.
            available_attributes (list): The list of attributes that can be used for splitting.
        Returns:
            list: A nested list representing the decision tree. The tree is built recursively and contains nodes
                  of the form ["Attribute", attribute_name], ["Value", attribute_value], and ["Leaf", class_label, 
                  class_count, total_count].
        The function follows these steps:
            1. Select an attribute to split on.
            2. Remove the selected attribute from the list of available attributes.
            3. Create a tree node for the selected attribute.
            4. Partition the instances based on the selected attribute's values.
            5. For each partition, check for base cases:
                - All class labels in the partition are the same: create a leaf node.
                - No more attributes to select: create a majority vote leaf node.
                - No more instances to partition: create a majority vote leaf node.
            6. If none of the base cases are met, recursively build the subtree for the partition.
            7. Append the subtree to the current tree node.
        """
        # basic approach (uses recursion!!):
        # select an attribute to split on
        split_attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(split_attribute) # can't split on this attribute again
        # in this subtree
        tree = ["Attribute", split_attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, split_attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value in sorted(partitions.keys()): # process in alphabetical order
            att_partition = partitions[att_value]
            value_subtree = ["Value", att_value]
            #    CASE 1: all class labels of the partition are the same
            # => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                class_label = att_partition[0][-1]
                class_count = len(att_partition)
                total = len(current_instances)
                leaf = ["Leaf", class_label, class_count, total]
                value_subtree.append(leaf)

            #    CASE 2: no more attributes to select (clash)
            # => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                class_labels = [row[-1] for row in current_instances]
                label_counts = {label: class_labels.count(label) for label in set(class_labels)}

                # Sort labels alphabetically, then by count (highest count first)
                sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
                class_label = sorted_labels[0][0]

                class_count = len(att_partition)
                total = len(current_instances)
                leaf = ["Leaf", class_label, class_count, total]
                value_subtree.append(leaf)

            #    CASE 3: no more instances to partition (empty partition)
            # => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                class_labels = [row[-1] for row in current_instances]
                label_counts = {label: class_labels.count(label) for label in set(class_labels)}

                # Sort labels alphabetically, then by count (highest count first)
                sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
                class_label = sorted_labels[0][0]
                class_count = len(att_partition)
                total = len(current_instances)
                leaf = ["Leaf", class_label, class_count, total]
                value_subtree.append(leaf)

            else:
                # none of base cases were true, recurse!!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                value_subtree.append(subtree)
            tree.append(value_subtree)
        return tree

class MyRandomForestClassifier:
    def __init__(self, n_classifiers=10, max_features=None, n_bootstrap=10):
        """
        Initializer for MyRandomForestClassifier.
        """
        self.n_classifiers = n_classifiers
        self.max_features = max_features
        self.n_bootstrap = n_bootstrap
        self.classifiers = []
        self.feature_subsets = []

    def compute_bootstrapped_sample(self, table):
        n = len(table)
        sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
        sample = [table[index] for index in sampled_indexes]
        out_of_bag_indexes = [index for index in range(n) if index not in sampled_indexes]
        out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
        return sample, out_of_bag_sample

    def compute_random_subset(self, values, num_values):
        values_copy = values[:]  
        np.random.shuffle(values_copy)
        return values_copy[:num_values]

    def fit(self, X_train, y_train):
        # Initialize the MyDecisionTreeClassifier
        tree_classifier = MyDecisionTreeClassifier()

        for _ in range(self.n_classifiers):
            # Create bootstrap sample
            X_sample, _ = self.compute_bootstrapped_sample(X_train)
            y_sample, _ = self.compute_bootstrapped_sample(y_train)

            # Select random feature subset
            feature_indices = self.compute_random_subset(list(range(len(X_train[0]))), self.max_features)
            self.feature_subsets.append(feature_indices)

            # Create a training subset based on the feature indices
            X_sample_sub = [[row[i] for i in feature_indices] for row in X_sample]

            # Train the decision tree classifier using the tdidt method
            tree_classifier.X_train = X_sample_sub
            tree_classifier.y_train = y_sample
            tree_classifier.header = [str(i) for i in feature_indices]  # Example: header based on indices
            tree_classifier.attribute_domains = {i: list(set([row[i] for row in X_sample])) for i in feature_indices}

            # Build the tree
            tree_classifier.tree = tree_classifier.tdidt(X_sample_sub, feature_indices[:])

            # Store the trained tree in the classifiers list
            self.classifiers.append(tree_classifier.tree)

    def _predict_tree(self, tree, row, feature_indices):
        # Traverse the decision tree
        while tree[0] == "Attribute":
            att_index = feature_indices[int(tree[1])] 
            value = row[att_index]

            matched = False
            for i in range(2, len(tree), 2):
                if tree[i][1] == value: 
                    tree = tree[i + 1]
                    matched = True
                    break
            if not matched:
                return None 
        return tree[1] 


    def predict(self, X):
        predictions = []
        for x in X:
            tree_predictions = []
            for tree, feature_indices in zip(self.classifiers, self.feature_subsets):
                prediction = self._predict_tree(tree, x, feature_indices)
                tree_predictions.append(prediction)

            # Majority vote 
            prediction = max(set(tree_predictions), key=tree_predictions.count)
            predictions.append(prediction)
        return predictions


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
