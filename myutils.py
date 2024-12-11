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


def get_rf_columns(column_indices, data):
    """
    Extracts specified columns from a dataset when fitting tree for random forest

    Args:
        column_indices (list of int): List of column indices to extract.
        data (list of list): The dataset from which to extract columns, where each inner list represents a row.

    Returns:
        list of list: A new dataset containing only the specified columns.
    """
    return [[row[idx] for idx in column_indices] for row in data]


# Functions for EDA
def get_column(table, header, col_name):
    """
    Extracts a column from a table based on the column name.
    Args:
        table (list of list): The table from which to extract the column, where each inner list represents a row.
        header (list): The header row containing the column names.
        col_name (str): The name of the column to extract.
    Returns:
        list: A list containing the values from the specified column.
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        col.append(row[col_index])

    return col

def discretize_bmi(bmi):
    """
    Discretizes a list of BMI values into categories: 'normal', 'overweight', and 'obese'.
    # < 25 normal
    # 25-30 overweight
    # > 30 is obese

    Parameters:
    bmi (list of float): A list of BMI values.

    Returns:
    list of str: A list of BMI categories corresponding to the input values.
    """
    for i in range(len(bmi)):
        if bmi[i] < 25.0:
            bmi[i] = 'normal'
        elif bmi[i] < 30.0:
            bmi[i] = 'overweight'
        else:
            bmi[i] = 'obese'
    return bmi

def discretize_physicalactivity(physical_activity):
    """
    Discretizes a list of physical activity levels into categories.
    # < 2 is inactive
    # 2-5 is active
    # > 5 is very active

    Parameters:
    physical_activity (list of float): A list of physical activity levels.

    Returns:
    list of str: A list where each physical activity level is replaced by a category:
                 'inactive' for levels less than 2.0,
                 'active' for levels between 2.0 and 5.0,
                 'very active' for levels greater than 5.0.
    """
    for i in range(len(physical_activity)):
        if physical_activity[i] < 2.0:
            physical_activity[i] = 'inactive'
        elif physical_activity[i] < 5.0:
            physical_activity[i] = 'active'
        else:
            physical_activity[i] = 'very active'
    return physical_activity

def discretize_lungfunction(lung_function):
    """
    Discretizes lung function values into categorical labels.

    # < 2.5 poor
    # >= 2.5 good

    Parameters:
    lung_function (list of float): A list of lung function values.

    Returns:
    list of str: A list of categorical labels ('poor' or 'good') corresponding to the input values.
    """
    for i in range(len(lung_function)):
        if lung_function[i] < 2.5:
            lung_function[i] = 'poor'
        else:
            lung_function[i] = 'good'
    return lung_function

def mean(column):
    """
    Calculate the mean of a list of numbers.

    Args:
        column (list): A list of numerical values.

    Returns:
        float: The mean of the numbers in the list, rounded to 2 decimal places.
    """
    return round(sum(column) / len(column), 2)

def stdev(column):
    """
    Calculate the standard deviation of a list of numbers.

    Parameters:
    column (list of float): A list of numerical values.

    Returns:
    float: The standard deviation of the input list, rounded to 2 decimal places.
    """
    mean_value = sum(column) / len(column)
    variance = sum((x - mean_value) ** 2 for x in column) / len(column)
    return round(variance ** 0.5, 2)

def mode(column):
    """
    Calculate the mode of a list of numbers.

    The mode is the value that appears most frequently in the list. 
    If there are multiple values with the same frequency, the function 
    returns the first one encountered. The result is rounded to 2 decimal places.

    Parameters:
    column (list): A list of numerical values.

    Returns:
    float: The mode of the list, rounded to 2 decimal places.
    """
    return round(max(set(column), key=column.count), 2)
