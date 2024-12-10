"""
Lauren Chin
CPSC 322, Fall 2024
Programming Assignment #6
11/06/2024

Description: This file many functions for
splitting data into training and testing sets,
and performing folds for these training
and testing sets.
This file has been updated to be able to calculate
precision, recall, and f1 scores for PA6.

Pylint score: 10/10
"""
import numpy as np # use numpy's random number generation

import myutils

# copy your myevaluation.py solution from PA5 here

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Set the random seed for reproducibility
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)

    # Calculate test size
    if isinstance(test_size, float):
        test_size = int(np.ceil(n_samples * test_size))
    elif isinstance(test_size, int):
        if test_size > n_samples:
            raise ValueError("test_size must be less than or equal to the number of samples.")

    # Shuffle the values if required
    if shuffle is True:
        myutils.randomize_in_place(X, y)

    # Calculate cutoff for train and test sets
    cutoff = n_samples - test_size

    # Create training and testing sets using the indices
    X_train = X[:cutoff]
    X_test = X[cutoff:]
    y_train = y[:cutoff]
    y_test = y[cutoff:]

    return X_train, X_test, y_train, y_test

def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    unique_labels = list(set(y))
    class_indices = {label: [] for label in unique_labels}
    for idx, label in enumerate(y):
        class_indices[label].append(idx)

    # Shuffle indices within each class if requested
    if shuffle:
        for indices in class_indices.values():
            randomize_in_place(indices, random_state=random_state)

    # Split indices into stratified folds
    folds = [[] for _ in range(n_splits)]
    for label, indices in class_indices.items():
        fold_sizes = [len(indices) // n_splits + (1 if i < len(indices) % n_splits else 0) for i in range(n_splits)]
        start_idx = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            folds[fold_idx].extend(indices[start_idx:start_idx + fold_size])
            start_idx += fold_size

    # Create train-test splits
    stratified_folds = []
    for fold_idx in range(n_splits):
        test_indices = folds[fold_idx]
        train_indices = [idx for i, fold in enumerate(folds) if i != fold_idx for idx in fold]
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n_samples = len(X)
    indices = list(range(n_samples))
    
    # Use randomize_in_place if shuffling is requested
    if shuffle:
        myutils.randomize_in_place(indices)
    # Calculate the fold sizes
    fold_sizes = [(n_samples // n_splits) + (1 if i < n_samples % n_splits else 0) for i in range(n_splits)]

    # Create folds
    folds = []
    current = 0
    for fold_size in fold_sizes:
        test_indices = indices[current:current + fold_size]
        train_indices = indices[:current] + indices[current + fold_size:]
        folds.append((train_indices, test_indices))
        current += fold_size

    return folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    # If n_samples is left to default, len(X)
    if n_samples is None:
        n_samples = len(X)

    if random_state is not None:
        np.random.seed(random_state)

    # Sample indices w/replacement
    sampled_indices = np.random.choice(len(X), size=n_samples, replace=True)
    X_sample = [X[i] for i in sampled_indices]

    # Determine out-of-bag indices
    out_of_bag_indices = np.setdiff1d(np.arange(len(X)), sampled_indices)
    X_out_of_bag = [X[i] for i in out_of_bag_indices]

    # Handle y if provided, because if None then only want to sample X
    if y is not None:
        y_sample = [y[i] for i in sampled_indices]
        y_out_of_bag = [y[i] for i in out_of_bag_indices]
    else:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Create mapping of labels to index
    label_index = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)

    # Initialize confusion matrix w/zeros
    matrix = [[0] * n_labels for _ in range(n_labels)]

    # Fill in confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in label_index and pred in label_index:
            matrix[label_index[true]][label_index[pred]] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    # Count number of correctly classified samples
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)

    # Calculate total samples
    n_samples = len(y_true)

    # Return normalized predictions if normalize=True
    if normalize:
        return correct_predictions / n_samples if n_samples > 0 else 0.0
    return correct_predictions

# PA6 work here
def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    # Use utils function to get confusion matrix values
    matrix_vals = myutils.gen_confusion_matrix_vals_w_pos(y_true, y_pred, labels, pos_label)
    t_p = matrix_vals[0] # first value in returned list is TP
    f_p = matrix_vals[1] # second value in returned list is FP

    # Compute precision
    precision = t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0.0

    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    # Use utils function to get confusion matrix values
    matrix_vals = myutils.gen_confusion_matrix_vals_w_pos(y_true, y_pred, labels, pos_label)
    t_p = matrix_vals[0] # first value in returned list is TP
    f_n = matrix_vals[2] # third value in returned list is FN

    # Compute recall
    recall = t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0.0

    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    # Use functions above to get precision and recall values
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    # Compute precision
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1_score
