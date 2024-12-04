import csv

import numpy as np # use numpy's random number generation

# From PA4 and PA5
def randomize_in_place(alist, parallel_list=None):
    """Randomizes alist elements in place. If a parallel_list is passed in as well,
    then it will be randomized parallel to alist elements so the elements
    in alist will still be aligned/paired with elements in parallel_list.

    Args:
        alist(list of elements): The list that will be randomized.
        parallel_list(list of elements): The parallel list to alist that will
            be randomized parallel to alist.

    """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]


def gen_confusion_matrix_vals_w_pos(y_true, y_pred, labels=None, pos_label=None):
    """This function generates a confusion matrix when 
    provided the positive class label.

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
        matrix_vals(list of int): Confusion matrix values [TP, FP, FN, TN]

    """
    # If no labels provided, use unique values in y_true
    if labels is None:
        labels = list(set(y_true))

    # Ensure labels are sorted
    labels.sort()

    # If pos_label is not provided, default to the first label in the list
    if pos_label is None:
        pos_label = labels[0]

    # Create mapping of labels to index
    label_index = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)

    # Initialize confusion matrix with zeros
    matrix = [[0] * n_labels for _ in range(n_labels)]

    # Fill in confusion matrix
    for true, pred in zip(y_true, y_pred):
        if true in label_index and pred in label_index:
            matrix[label_index[true]][label_index[pred]] += 1

    # If pos_label is provided, calculate TP, FP, FN, TN
    if pos_label is not None:
        # Locate the row and column corresponding to the positive class
        pos_idx = label_index[pos_label]

        # Extract the confusion matrix values for the positive class
        t_p = matrix[pos_idx][pos_idx]  # True positives
        f_p = sum(matrix[i][pos_idx] for i in range(n_labels)) - t_p  # False positives
        f_n = sum(matrix[pos_idx][i] for i in range(n_labels)) - t_p  # False negatives
        t_n = sum(sum(row) for row in matrix) - t_p - f_p - f_n  # True negatives

    # Organize confusion matrix values to a list to return
    matrix_vals = [t_p, f_p, f_n, t_n]
    return matrix_vals

def load(name):
    """loads data from tv_shows.csv

    Args:
        name of the csv file

    Returns:
        list of list: The table loaded in
    """

    table = []
    with open(name, 'r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        for row in reader:
            table.append(row)

    infile.close()
    return table


def remove_cols(table, headers, col_names):
    col_indexes = [i for i in range(len(headers)) if headers[i] in col_names]

    # print(col_indexes)

    for row in table:  
        for i in sorted(col_indexes, reverse=True):
            del row[i]


    for col_name in col_names:
        headers.remove(col_name)


    return table, headers

# Functions for EDA
def get_column(table, header, col_name):
    """
    returns a column in the table
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        col.append(row[col_index])

    return col