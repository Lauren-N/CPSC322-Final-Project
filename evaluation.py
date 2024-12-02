import utils       


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
        utils.randomize_in_place(indices)
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
