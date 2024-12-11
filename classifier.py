import numpy as np
import myevaluation
import myutils
from collections import Counter

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

        att_index = self.header.index(attribute)
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



    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.
        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.header = [f"att{i}" for i in range(len(X_train[0]))]
        self.attribute_domains = {
            self.header[i]: list(set(row[i] for row in X_train))
            for i in range(len(X_train[0]))
        }
        # print(self.attribute_domains)

        combined_data = [x + [y] for x, y in zip(X_train, y_train)]
        self.tree = self.tdidt(combined_data, self.header[:])

    def tdidt_predict(self, tree, instance, header):
        """
        Predict the class label for a given instance using a decision tree.
        Parameters:
        tree (list): The decision tree represented as a nested list.
        instance (list): The instance to classify.
        header (list): The list of attribute names corresponding to the instance.
        Returns:
        The predicted class label for the given instance.
        """

        # THINK ISSUE IS HERE

        # base case: we are at a leaf node and can return the class prediction
        info_type = tree[0] # "Leaf" or "Attribute"
        if info_type == "Leaf":
            return tree[1] # class label

        att_index = header.index(tree[1])
        for i in range(2, len(tree)):
            value_list = tree[i]
            # do we have a match with instance for this attribute?

            # need to find a way to create a cutoff value
            if value_list[1] == instance[att_index]:
                return self.tdidt_predict(value_list[2], instance, header)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []

        for instance in X_test:
            y_pred.append(self.tdidt_predict(self.tree, instance, self.header))
        # print(y_pred)
        return y_pred

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.
        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = self.header

        def traverse(tree, conditions):
            if tree[0] == "Leaf":
                # Base case: We've reached a leaf
                label = tree[1]
                count = tree[2]
                total = tree[3]
                rule = " AND ".join(conditions)
                print(f"IF {rule} THEN {class_name} = {label} [{count}/{total}]")
            elif tree[0] == "Attribute":
                # Recursive case: Traverse each value subtree
                attribute = attribute_names[self.header.index(tree[1])]
                for value_subtree in tree[2:]:
                    value = value_subtree[1]
                    traverse(value_subtree[2], conditions + [f"{attribute} == {value}"])

        traverse(self.tree, [])

class MyRandomForestClassifier:
    def __init__(self, N, M, F, seed=0):
        """
        Initializes the classifier with the given parameters.

        Parameters:
        N (int): The number of trees
        M (int):  M most accurate trees
        F (int): F attributes to consider
        seed (int, optional): The seed for random number generation. Default is 0.
        """
        self.X_train = None
        self.y_train = None
        self.N = N
        self.M = M
        self.F = F
        self.seed = seed
        self.trees = None

    # ginas function
    def compute_bootstrapped_sample(self, table):
        """
        Generates a bootstrapped sample and its corresponding out-of-bag sample from the given table.
        Taken from the lecture notes.

        Parameters:
        table (list): The input data table from which the samples are to be generated.

        Returns:
        tuple: A tuple containing two lists:
            - sample (list): The bootstrapped sample generated by randomly sampling with replacement from the input table.
            - out_of_bag_sample (list): The out-of-bag sample consisting of elements from the input table that were not included in the bootstrapped sample.
        """
        n = len(table)
        sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
        sample = [table[index] for index in sampled_indexes]
        out_of_bag_indexes = [index for index in list(range(n)) if index not in sampled_indexes]
        out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
        return sample, out_of_bag_sample

    # ginas function
    def compute_random_subset(self, values, num_values):
        """
        Generates a random subset of the given list of values.
        Taken from lecture notes.

        Parameters:
        values (list): The list of values from which to generate the subset.
        num_values (int): The number of values to include in the subset.

        Returns:
        list: A list containing `num_values` randomly selected elements from the input list.
        """
        values_copy = values[:] # shallow copy
        np.random.shuffle(values_copy) # in place shuffle
        return values_copy[:num_values]

    def fit(self, X_train, y_train):
        """
        Fit the Random Forest classifier on the training data.
        Parameters:
        X_train (list of lists): The training input samples.
        y_train (list): The target values (class labels) for the training input samples.
        This method performs the following steps:
        1. Sets the training data as object variables.
        2. Generates stratified k-fold splits of the training data.
        3. Selects a fold to use for fitting.
        4. Creates bootstrap samples and random subsets of features.
        5. Trains multiple decision trees using the bootstrapped samples and subsets of features.
        6. Computes the accuracy of each tree on the validation set.
        7. Selects the top M most accurate trees and stores them as object variables.
        """
        # set object variables
        self.X_train = X_train
        self.y_train = y_train
        N = self.N
        M = self.M
        F = self.F
        np.random.seed(self.seed)

        # generating folds to get a random stratified sample!
        folds = myevaluation.stratified_kfold_split(X_train, y_train, n_splits=10, shuffle=True, random_state=0)

        # pick fold in list to use for fit and for validating accuracies among trees
        fold=folds[2]
        
        # data for bootstraping
        X = [X_train[x] for x in fold[0]]
        y = [y_train[x] for x in fold[0]]

        # list to hold all trees produced
        all_trees = []

        # to hold accuracies for later
        accuracies = []

        # generating random trees using bootstrap samples and random subsets
        for n in range(N):
            # bootstrapping using ginas function
            X_train, X_test = self.compute_bootstrapped_sample(X)
            y_train, y_test = self.compute_bootstrapped_sample(y)

            # randomly sampling F features to create trees
            attributes = self.compute_random_subset(list(range(len(X_train[0]))), F)
            attributes.sort()

            # create decision tree with random subset of features and bootstrapped samples
            tree = MyDecisionTreeClassifier()

            # getting only columns in random subset to fit in decision trees classifier function
            tree.fit(myutils.get_rf_columns(attributes, X_train), y_train)

            # appending generated tree to list
            all_trees.append([tree.tree, attributes])

            # validating tree using stratified sample to compute accuracy to get best tree
            y_pred=tree.predict(myutils.get_rf_columns(attributes, X_test))

            # computing accuracy
            accuracy = myevaluation.accuracy_score(y_test, y_pred)

            # adding accuracy to list
            accuracies.append((accuracy, n))

        
        # finding the most accurate tree
        accuracies.sort(key=lambda x: x[0], reverse=True)

        # sorting trees by accuracy
        sorted_trees = [accuracy[1] for accuracy in accuracies[:M]]

        # setting class variable to hold trees
        self.trees = [all_trees[i] for i in sorted_trees]


    def predict(self, X_test): 
        """
        Predicts the class labels for the given test data using the trained random forest classifier.
        Parameters:
        X_test (list of list of obj): The test data to predict, where each inner list represents a single instance.
        Returns:
        list of obj: The predicted class labels for each instance in the test data.
        """
        # holds predictions to return
        predictions = []

        # predictions from all trees
        for tree, feature_subset in self.trees:
            # setting up tree
            pred_tree = MyDecisionTreeClassifier()

            # setting tree
            pred_tree.tree = tree

            # setting header and attribute domains
            pred_tree.header = [f"att{i}" for i in feature_subset]
            pred_tree.attribute_domains = {
                pred_tree.header[i]: list(set(row[i] for row in self.X_train))
                for i in range(len(pred_tree.header))
            }

            # getting only columns in random subset to fit in decision trees classifier function
            feature_test = myutils.get_rf_columns(feature_subset, X_test)

            # getting predictions
            predictions.append(pred_tree.predict(feature_test))


        # majority vote for each instance
        y_pred = []

        # getting majority vote for each instance
        for i in range(len(X_test)):
            # getting all predictions for each instance
            votes = [preds[i] for preds in predictions]

            # getting majority vote
            majority_vote = Counter(votes).most_common(1)[0][0]

            # appending majority vote to list
            y_pred.append(majority_vote)

        # returning predictions
        return y_pred


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
    
