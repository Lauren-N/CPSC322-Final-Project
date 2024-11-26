import csv


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

def compute_slope_intercept(x,y):
    n = len(x)
    meanx = sum(x) / len(x)
    meany = sum(y) / len(y)
    
    numerator = sum([(x[i] - meanx) * (y[i] - meany) for i in range(n)])
    denominator = sum([(x[i] - meanx) ** 2 for i in range(n)])

    m = numerator/denominator
    # y = mx + b -> b = y - mx
    b = meany - m * meanx
    return m,b