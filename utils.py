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