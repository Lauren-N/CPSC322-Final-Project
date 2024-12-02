"""
Lauren Chin
CPSC 322, Fall 2024
Programming Assignment #5
9/29/2024

Description: This file creates/fills in the MyPyTable methods
in the MyPyTable class. It creates methods like computing
summary statistics, performing inner and outer joins, 
removing missing values, and more. (Copied over from PA2)
"""
import copy
import csv
from tabulate import tabulate

# copy your mypytable.py solution from PA2-PA5 here

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        n_rows = len(self.data) # n num of lists in data list (rows)
        m_cols = len(self.column_names) # m num of elements in each list in data list (cols)
        return n_rows, m_cols

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        if  isinstance(col_identifier, int):
            # then is column index
            col_index = col_identifier
        else:
            # then is string for column name
            for i in range(len(self.column_names)) :
                if self.column_names[i] == col_identifier:
                    # then i is index
                    col_index = i
                #else then continue looking through column_names
        col = []
        if include_missing_values:
            # if include_missing_values is True then add all rows
            for row in self.data:
                col.append(row[col_index])
        else:
            # then include_missing_values if False
            for row in self.data:
                if row[col_index] != "":
                    # only append if not a missing value
                    col.append(row[col_index])

        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for i in range(len(self.data)):
            for j in range(len(self.column_names)):
                # goes through each element in table
                try:
                    self.data[i][j] = float(self.data[i][j])
                except ValueError:
                    #nothing happens to self.data
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        # reverse sorting index values and dropping duplicates
        reverse_indexes = sorted(set(row_indexes_to_drop), reverse=True)
        # dropping rows starting at largest index
        for index in reverse_indexes:
            if 0 <= index < len(self.data):
                # making sure is valid index
                self.data.pop(index)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        header = []
        table = []
        # 1. open the file
        infile = open(filename, "r")
        # 2. process the file
        reader = csv.reader(infile)
        index = 0
        for row in reader:
            if index == 0:
                # first row so header
                header = row
            else:
                table.append(row)
            index += 1
        # 3. close the file
        infile.close()
        # set the attributes
        self.column_names = header
        self.data = table
        # call convert_to_numeric()
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        outfile = open(filename, "w")
        writer = csv.writer(outfile)
        writer.writerow(self.column_names)
        writer.writerows(self.data)
        outfile.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        key_indexes = []
        for col in key_column_names:
            key_indexes.append(self.column_names.index(col))

        duplicate_indexes = []
        checked_rows = set()

        for i, row in enumerate(self.data):
            # create a key_row with values from the key_column_names
            key_row = tuple(row[index] for index in key_indexes)

            if key_row in checked_rows:
                # is duplicate to add index
                duplicate_indexes.append(i)
            else:
                # otherwise add to checked_rows
                checked_rows.add(key_row)

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        final_table = []
        for i in range(len(self.data)):
            missing = False
            for j in range(len(self.column_names)):
                element = self.data[i][j]
                # if (element == "") or (element == "NA"):
                if (element in ("", "NA")):
                    missing = True
            if missing is False:
                final_table.append(self.data[i])
        self.data = final_table

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        # calculating average of column
        col_index = (self.column_names).index(col_name)
        col_avg = 0
        count = 0
        for row in self.data:
            if (row[col_index] != "") and (row[col_index] != "NA"):
                col_avg += row[col_index]
                count += 1
        col_avg = col_avg/count
        # checking missing values for specified column and filling in
        for i in range(len(self.data)):
            # setting element to average if missing/NA
            element = self.data[i][col_index]
            # if (element == "") or (element == "NA"):
            if (element in ("", "NA")):
                self.data[i][col_index] = col_avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        summary_statistics = []
        summary_header = ["attribute", "min", "max", "mid", "avg", "median"]

        for col in col_names:
            col_index = (self.column_names).index(col)
            col_values = []
            # populate col_values
            for row in self.data:
                if row[col_index] != "":
                    col_values.append(float(row[col_index]))

            # if col_values is empty then continue
            if len(col_values) == 0:
                continue
            # calculate statistics using list build-ins
            col_min = min(col_values)
            col_max = max(col_values)
            col_avg = sum(col_values) / len(col_values)
            col_mid = (col_min + col_max) / 2
            sorted_values = sorted(col_values)
            if len(sorted_values) % 2 != 0:
                # if odd, return the middle value
                col_median = sorted_values[len(sorted_values) // 2]
            else:
                # if even, return the average of the two middle values
                mid_index = len(sorted_values) // 2
                col_median = (sorted_values[mid_index - 1] + sorted_values[mid_index]) / 2

            # append statistics for current column
            summary_statistics.append([
                col, col_min, col_max, col_mid, col_avg, col_median
            ])
        # create MyPyTable object and return
        summary_table = MyPyTable(summary_header, summary_statistics)
        return summary_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        # find indexes of key columns in self table
        common_key_indexes_self = []
        for col in key_column_names:
            if col in self.column_names:
                common_key_indexes_self.append(self.column_names.index(col))

        # find indexes of key columns in other_table
        common_key_indexes_other = []
        for col in key_column_names:
            if col in other_table.column_names:
                common_key_indexes_other.append(other_table.column_names.index(col))

        # combine column headers
        # first copy over self's column names
        combined_header = self.column_names.copy()
        # now add other_table's column names
        for col in other_table.column_names:
            if col not in combined_header:
                combined_header.append(col)

        joined_data = []

        # perform inner join
        for row_self in self.data:
            for row_other in other_table.data:
                # check if keys match
                if all(row_self[index] == row_other[other_table.column_names.index(key_column_names[i])]
                    for i, index in enumerate(common_key_indexes_self)):
                    # combine rows but avoid duplicating key columns
                    joined_row = row_self[:]
                    for i, elem in enumerate(row_other):
                        if i not in common_key_indexes_other:
                            joined_row.append(elem)
                    joined_data.append(joined_row)

        return MyPyTable(combined_header, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # find indexes of key columns in self table
        common_key_indexes_self = []
        for col in key_column_names:
            if col in self.column_names:
                common_key_indexes_self.append(self.column_names.index(col))

        # find indexes of key columns in other_table
        common_key_indexes_other = []
        for col in key_column_names:
            if col in other_table.column_names:
                common_key_indexes_other.append(other_table.column_names.index(col))

        # combine column headers
        # first copy over self's column names
        combined_header = self.column_names.copy()
        # now add other_table's column names
        for col in other_table.column_names:
            if col not in combined_header:
                combined_header.append(col)

        joined_data = []
        processed_keys = set()

        # outer join
        for row_self in self.data:
            key_self = tuple(row_self[index] for index in common_key_indexes_self)
            key_match = False

            for row_other in other_table.data:
                key_other = tuple(row_other[index] for index in common_key_indexes_other)
                # check if keys match
                if key_self == key_other:
                    key_match = True
                    # combine rows but avoid duplicating key columns
                    joined_row = row_self[:]
                    for i, elem in enumerate(row_other):
                        if i not in common_key_indexes_other:
                            joined_row.append(elem)
                    joined_data.append(joined_row)

            if not key_match:
                # if no match, create row with "NA" for other_table's columns
                joined_row = row_self[:] + ["NA"] * (len(combined_header) - len(row_self))
                joined_data.append(joined_row)

            processed_keys.add(key_self)

        # handle rows in other table that didn't match
        for row_other in other_table.data:
            key_other = tuple(row_other[index] for index in common_key_indexes_other)
            if key_other not in processed_keys:
                # create a row with "NA" for self's columns
                joined_row = ["NA"] * len(self.column_names)

                # fill in values from other_table
                for i, col_name in enumerate(other_table.column_names):
                    if col_name not in self.column_names:
                        # place value in correct position
                        joined_row.append(row_other[i])
                    else:
                        # if column is also in self, place correctly
                        index_in_combined_header = combined_header.index(col_name)
                        joined_row[index_in_combined_header] = row_other[i]
                # append the joined row to the data
                joined_data.append(joined_row)
        return MyPyTable(combined_header, joined_data)

    def extract_key_from_row(self, row, header, key):
        """Returns the values for the key from the row passed in
        
        Args:
            row (list): row of data
            header (list): list of column names
            key (str): key

        Returns:
            value (whatever value of element in row): Value for key from row passed in
        """
        key_index = header.index(key)
        value = row[key_index]
        return value
