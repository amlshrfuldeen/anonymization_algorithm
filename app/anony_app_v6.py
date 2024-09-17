# Standard library imports
import hashlib
import random
import time
import re

# Third-party imports
import numpy as np
import pandas as pd
import pyodbc
import psutil
import seaborn as sns
import streamlit as st


def extract_tables_and_columns(sql_query):
    """
    Extracts table names, column names, and join conditions from an SQL query.

    This function uses regular expressions to extract:
    - Table names from SQL clauses (e.g., FROM, JOIN).
    - Column names from the SELECT clause.
    - Join conditions from JOIN clauses.

    Parameters:
    ----------
    sql_query : str
        The SQL query string to extract tables, columns, and join conditions from.

    Output:
    -------
    table_columns_mapping = {
        'table1': ['col1'],
        'table2': ['col2']
    }
    tables_columns_list = ['t1.col1', 't2.col2']
    join_columns_mapping = [
        {'left_col': 't1.col1', 'right_col': 't2.col1'}
    ]
    """
    # Regular expressions to match table names and column names
    table_pattern = re.compile(r"(?:FROM|JOIN)\s+(\w+)", re.IGNORECASE)
    column_pattern = re.compile(r"SELECT\s+(.*?)\s+FROM", re.IGNORECASE | re.DOTALL)
    join_condition_pattern = re.compile(
        r"JOIN\s+\w+\s+ON\s+(.+?)(?=\s+(?:JOIN|WHERE|GROUP|ORDER|$))", re.IGNORECASE
    )

    # Extract table names
    tables = []
    for match in table_pattern.finditer(sql_query):
        table_name = match.group(1)
        if table_name:
            tables.append(table_name.strip())

    # Extract column names
    columns_str = column_pattern.search(sql_query)
    columns = []
    if columns_str:
        columns_str = columns_str.group(1)
        if columns_str.strip() == "*":
            # If SELECT *, map all columns from each table to '*'
            columns = ["*"]
        else:
            # Otherwise, split the column list
            columns = [
                col.strip() for col in re.split(r",\s*(?![^()]*\))", columns_str)
            ]

    # Map columns to tables
    tables_columns_list = []
    table_columns_mapping = {table: [] for table in tables}
    if tables and columns:
        if "*" in columns:
            # If '*' is used, map '*' to all tables
            for table in tables:
                table_columns_mapping[table].append("*")
        else:
            for column in columns:
                # Check if the column is qualified with a table name
                if "." in column:
                    table_name, column_name = column.split(".")
                    table_name = table_name.strip()  # to delete any spaces
                    column_name = column_name.strip()  # to delete any spaces
                    if table_name in table_columns_mapping:
                        table_columns_mapping[table_name].append(column_name)
                        tables_columns_list.append(table_name + "." + column_name)
                else:
                    # If no table name is specified, map the column to all tables
                    for table in tables:
                        table_columns_mapping[table].append(column)
                        tables_columns_list.append(column)

    # Extract join conditions
    join_columns_mapping = []
    for match in join_condition_pattern.finditer(sql_query):
        condition = match.group(1)
        conditions = re.split(r"\s+AND\s+", condition, flags=re.IGNORECASE)
        for cond in conditions:
            # Split on '=' to get the two columns being joined
            if "=" in cond:
                left_col, right_col = [col.strip() for col in cond.split("=")]
                join_columns_mapping.append(
                    {
                        "left_col": left_col.strip(),
                        "right_col": right_col.strip(),
                    }
                )

    return table_columns_mapping, tables_columns_list, join_columns_mapping


def validate_columns_in_tables(cursor, table_columns_mapping):
    """
    Validates if the specified columns exist in the corresponding tables from the SQL Server.

    This function checks if the columns provided in `table_columns_mapping` exist in the tables
    in the SQL Server database. If a column is '*', it retrieves all columns from the table.
    It returns a dictionary containing only the valid columns for each table.

    Parameters:
    ----------
    cursor : pyodbc.Cursor
        A cursor object to execute SQL queries on the SQL Server database.
    table_columns_mapping : dict
        A dictionary where keys are table names and values are lists of columns
        that need to be validated.

    Output:
    -------
    valid_columns = {
        'table1': ['col1', 'col2'],
        'table2': ['col1', 'col2', 'col3']  # All columns if '*' is used
    }
    """
    valid_table_columns_mapping = {}

    for table, columns in table_columns_mapping.items():
        valid_columns = []

        # Fetch all columns from the current table in SQL Server
        cursor.execute(
            f"""
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table}'
        """
        )
        table_columns = [row[0] for row in cursor.fetchall()]

        for column in columns:
            if column == "*":
                # If column is '*', add all table columns
                valid_columns.extend(table_columns)
            elif column in table_columns:
                # Validate if the column exists in the table
                valid_columns.append(column)

        if valid_columns:
            valid_table_columns_mapping[table] = valid_columns

    return valid_table_columns_mapping


def display_all_columns(columns):
    """
    Display the list of available columns with their index numbers.

    """
    # Display the columns with their index numbers
    st.write("Available columns:")
    for i, column in enumerate(columns):
        st.write(
            f"{i + 1}. {column}"
        )  # i + 1 because we want to show 1-based indexing to the user


def select_columns_to_modify(tables_columns_list, selected_indices_input):
    """
    Prompts the user to select columns for modification based on input indices.

    This function takes a list of table columns and user input representing the indices
    of columns to be selected. It validates the input and returns a list of the selected
    columns. If a table and column are provided in the form of 'table.column', it extracts
    only the column name.

    Parameters:
    ----------
    tables_columns_list : list
        A list of column names or 'table.column' formatted strings representing
        the columns available for selection.
    selected_indices_input : str
        A comma-separated string of indices input by the user, which represent the
        positions of columns in the `tables_columns_list`.

    Output:
    -------
    selected_columns = ['col1', 'col2']
    """
    # display_all_columns(tables_columns_list)
    selected_columns = []

    try:
        # Convert the input into a list of integers
        selected_indices = [int(i) for i in selected_indices_input.split(",")]

        # Validate the indices
        if all(1 <= i <= len(tables_columns_list) for i in selected_indices):
            if "." in tables_columns_list:  # Ensure there are table names in the list
                selected_columns = [
                    tables_columns_list[i - 1].split(".")[1] for i in selected_indices
                ]  # Subtract 1 to convert to 0-based index
            else:
                selected_columns = [
                    tables_columns_list[i - 1] for i in selected_indices
                ]
        else:
            st.error(
                f"Error: Please enter numbers between 1 and {len(tables_columns_list)}."
            )
    except ValueError:
        st.error("Error: Please enter valid numbers.")

    # Select the columns based on the user's choices
    st.write("selected_columns", selected_columns)
    return selected_columns


def modify_column_names(columns):
    """
    Appends a suffix ("_noisy") to the selected column names.

    This function takes a list of column names and returns a dictionary where
    the original column names are the keys, and the modified column names (with
    the "_noisy" suffix) are the values.

    Parameters:
    ----------
    columns : list
        A list of column names to be modified.

    Output:
    -------
    modified_columns = {
        'col1': 'col1_noisy',
        'col2': 'col2_noisy'
    }
    """
    modified_columns = {}
    for column in columns:
        modified_name = (
            f"{column}_noisy"  # Example: Add "_noisy" at the end of the name
        )
        modified_columns[column] = modified_name
    return modified_columns


def get_column_type(cursor, table, column):
    """
    Retrieves the data type, length, and key constraints (primary and foreign) of a specified column
    in a given table.

    This function queries the SQL Server database to get the data type, length, and checks
    whether the column is a primary key or a foreign key.

    Parameters:
    ----------
    cursor : pyodbc.Cursor
        A cursor object to execute SQL queries on the SQL Server database.
    table : str
        The name of the table where the column is located.
    column : str
        The name of the column to retrieve the data type and constraints for.

    Output:
    -------
    data_type = 'int'
    char_length = '32'
    is_primary_key = True
    is_foreign_key = False
    """
    # get column data type
    cursor.execute(
        f"""
          SELECT DATA_TYPE,
          CASE
             WHEN DATA_TYPE IN ('char', 'varchar', 'nchar', 'nvarchar') THEN
                 CAST(CHARACTER_MAXIMUM_LENGTH AS VARCHAR)
             WHEN DATA_TYPE IN ('decimal', 'numeric') THEN
                 CAST(NUMERIC_PRECISION AS VARCHAR) + ',' + CAST(NUMERIC_SCALE AS VARCHAR)
             WHEN DATA_TYPE IN ('float', 'real') THEN
                 CAST(NUMERIC_PRECISION AS VARCHAR)
             WHEN DATA_TYPE IN ('int', 'smallint', 'tinyint', 'bigint') THEN
                CAST(
                  CASE
                      WHEN DATA_TYPE = 'int' THEN 32
                      WHEN DATA_TYPE = 'smallint' THEN 16
                      WHEN DATA_TYPE = 'tinyint' THEN 8
                      WHEN DATA_TYPE = 'bigint' THEN 64
                  END
                AS VARCHAR)
              ELSE NULL
          END AS Column_Length
          FROM INFORMATION_SCHEMA.COLUMNS
          WHERE TABLE_NAME = '{table}' AND COLUMN_NAME = '{column}'
      """
    )
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"Column '{column}' not found in table '{table}'")
    data_type, char_length = result

    # Query to check if the column is a primary key
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME = ? AND COLUMN_NAME = ?
    """,
        table,
        column,
    )
    is_primary_key = cursor.fetchone() is not None

    # Query to check if the column is a foreign key
    cursor.execute(
        """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_NAME = ? AND COLUMN_NAME = ?
        AND OBJECTPROPERTY(OBJECT_ID(CONSTRAINT_NAME), 'IsForeignKey') = 1
    """,
        table,
        column,
    )
    is_foreign_key = cursor.fetchone() is not None

    return data_type, char_length, is_primary_key, is_foreign_key


def check_and_create_columns(conn, table_name, column_name, column_type, char_length):
    """
    Checks if a column exists in a specified table, and if it doesn't, the column is created.

    This function queries the SQL Server database to check if a given column exists in a specified
    table. If the column does not exist, the function creates it using the specified column type
    and, optionally, a character length (for VARCHAR-like data types).

    Parameters:
    ----------
    conn : pyodbc.Connection
        A connection object to the SQL Server database.
    table_name : str
        The name of the table to check for the column.
    column_name : str
        The name of the column to check or create.
    column_type : str
        The data type of the column (e.g., 'VARCHAR', 'INT').
    char_length : int or None
        The character length for data types that require it (e.g., VARCHAR). If the data type
        doesn't require a length, set it to None.

    Output:
    -------
    If the column does not exist:
    "Column 'email' added to table 'users'."

    If the column already exists:
    "Column 'email' already exists in table 'users'."
    """
    cursor = conn.cursor()

    # SQL query to check if the column already exists in the specified table
    query = f"""
    SELECT COUNT(*)
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{table_name}' AND COLUMN_NAME = '{column_name}'
    """
    cursor.execute(query)
    result = cursor.fetchone()

    # If the column does not exist (i.e., COUNT(*) returned 0)
    if result[0] == 0:

        # If the column has a specified character length (for types like VARCHAR)
        if char_length is not None:
            # Append the length to the column type, e.g., VARCHAR(255)
            column_type = f"{column_type}({char_length})"
        else:
            # Use the column type without any additional length, e.g., INT
            column_type = f"{column_type}"

        # SQL query to add the new column to the table
        query = f"ALTER TABLE {table_name} ADD {column_name} {column_type}"
        cursor.execute(query)
        conn.commit()

        # Print a message indicating that the column was successfully added
        st.write(f"Column '{column_name}' added to table '{table_name}'.")

    else:
        # If the column already exists, print a message indicating so
        st.write(f"Column '{column_name}' already exists in table '{table_name}'.")

    cursor.close()


def add_modified_columns_to_sql_query(sql_query, columns, suffix):
    """
    Modifies a SQL query by appending a suffix to the specified column names.

    This function takes an SQL query as input, identifies column names that are present in the
    provided list, and appends a specified suffix to these column names. The function uses
    regular expressions to match column names and replaces them accordingly.

    Parameters:
    ----------
    sql_query : str
        The original SQL query in which column names will be modified.
    columns : list of str
        A list of column names to which the suffix will be appended. The column names can
        be specified with or without table prefixes.
    suffix : str
        The suffix to append to each matched column name.

    # Output: "SELECT id_modified, name_modified FROM users WHERE age > 30"
    """
    # Regular expression to match column names
    column_pattern = re.compile(r"\b(\w+\.\w+|\w+)\b")

    # Function to replace column names with the suffixed version
    def replace_column(match):
        # Extract the matched column name from the regex match object
        column_name = match.group(1)

        # Check if the column name is in the provided list (with or without table prefix)
        for col in columns:
            # Check if the column name ends with the provided column name
            if column_name.endswith(col):
                # Extract the table prefix from the column name, if it exists
                table_prefix = column_name[: -len(col)]
                # Return the column name with the specified suffix added
                return f"{table_prefix}{col}{suffix}"
        return column_name  # Return the original column name if no match

    # Replace column names in the SQL query
    new_sql_query = column_pattern.sub(replace_column, sql_query)

    return new_sql_query


def update_column(cursor, table, column, noise_value, original_value):
    """
    Updates a noisy column in the specified table based on a condition.

    This function updates a column in a SQL Server table by setting its value to `noise_value`
    where another specified column matches `original_value`. The noisy column is created by
    appending "_noisy" to the original column name.

    Parameters:
    ----------
    cursor : pyodbc.Cursor
        A cursor object to execute SQL queries on the SQL Server database.
    table : str
        The name of the table where the column will be updated.
    column : str
        The name of the column to be updated with noise values.
    noise_value : str
        The new value to set in the noisy column.
    original_value : str
        The value in the original column that is used to identify the rows to update.

    Output:
    -------
    Updates the 'email_noisy' column in the 'users' table
    where the 'email' column equals 'example@example.com'.
    """
    noisy_column = column + "_noisy"
    # Construct the SQL UPDATE statement with placeholders
    update_query = f"""
        UPDATE {table}
        SET {noisy_column} = ?
        WHERE {column} = ?
    """
    # Execute the query with parameterized values
    cursor.execute(update_query, (noise_value, original_value))
    cursor.connection.commit()


def apply_anonymization_tech_to_column(
    conn, table, column, is_primary_key, is_foreign_key, column_type
):
    """
    Applies appropriate anonymization techniques to the specified column
    in a table based on its data type.

    This function adds noise or applies hashing to a column in a SQL Server table.
    The type of anonymization technique
    applied depends on whether the column is a primary key, foreign key,
    or contains integer, float, decimal, string, or date values.

    Parameters:
    ----------
    conn : pyodbc.Connection
        A connection object to interact with the SQL Server database.
    table : str
        The name of the table containing the column to be modified.
    column : str
        The name of the column to which anonymization will be applied.
    is_primary_key : bool
        Indicates if the column is a primary key.
    is_foreign_key : bool
        Indicates if the column is a foreign key.
    column_type : str
        The data type of the column, which determines the type of anonymization to apply
        (e.g., 'int', 'float', 'varchar', 'date').

    Output:
    -------
    Updates the 'age' column in the 'users' table with integer noise or
    other specified anonymization techniques.
    """
    cursor = conn.cursor()
    try:
        # Fetch distinct values from the column to generate consistent noise
        cursor.execute(
            f"""
            SELECT {column}
            FROM {table}
            WHERE {column} IS NOT NULL
        """
        )
        data = cursor.fetchall()

        if is_foreign_key or is_primary_key:

            encoded_data_key = encode_data(data)
            for value, noisy_value in zip(data, encoded_data_key):
                update_column(cursor, table, column, noisy_value, value[0])

        elif column_type in ["int", "bigint", "smallint", "tinyint", "numeric"]:

            encoded_data_int = add_int_noise(data, 10)
            for value, noisy_value in zip(data, encoded_data_int):
                update_column(cursor, table, column, noisy_value, value[0])

        elif column_type in ["decimal", "float"]:

            encoded_data_decimal = add_decimal_noise(data, 10)
            for value, noisy_value in zip(data, encoded_data_decimal):
                update_column(cursor, table, column, noisy_value, value[0])

        elif column_type in ["nvarchar", "char", "varchar", "text", "ntext"]:

            encoded_data_char = encode_data(data)
            for value, noisy_value in zip(data, encoded_data_char):
                update_column(cursor, table, column, noisy_value, value[0])

        elif column_type in [
            "date",
            "time",
            "datetime",
            "datetime2",
            "smalldatetime",
            "datetimeoffset",
        ]:

            encoded_data_date = add_noise_to_dates(data)
            for value, noisy_value in zip(data, encoded_data_date):
                update_column(cursor, table, column, noisy_value, value[0])

        else:
            st.error("The data type is not supported")

    except Exception as e:
        st.error(f"Error in apply_anonymization_tech_to_column: {e}")


def make_number_noise(noise_level):
    """
    Generates a random noise value to be added to the data.
    The noise is determined based on a Gaussian distribution.

    Parameters:
    ----------
    value : float
        The original data value to which noise will be added.
    noise_level : float
        The standard deviation of the noise, which determines the spread of the distribution.

    Returns:
    -------
    float
        A noise value generated based on a Gaussian distribution.

    """
    mean = 0  # Mean of the noise distribution
    std_dev = noise_level  # Standard deviation of the noise distribution

    # Generate Gaussian noise with mean 0 and std_dev as noise_level
    noise = np.random.normal(mean, std_dev)

    return noise


def add_int_noise(int_column, noise_level):
    """
    Adds random noise to integer data values.

    Parameters:
    ----------
    int_column : list or numpy array
        A list or array of integer values to which noise will be added.
    noise_level : float
        The standard deviation of the noise distribution.

    Returns:
    -------
    numpy array
        An array of noisy integer values.

    """
    # Convert the input column to a NumPy array and flatten it
    data = np.array(int_column).flatten()

    # Identify unique values in the data
    unique_values = np.unique(data)

    # Map each unique value to a noise value
    try:
        value_to_noise = {
            value: make_number_noise(noise_level) for value in unique_values
        }
    except Exception as e:
        st.error(f"Error in generating noise values: {e}")
        return np.array([])  # Return an empty array in case of error

    # Create an array of noise values corresponding to the unique values
    value_to_noise_array = np.array([value_to_noise[val] for val in unique_values])

    # Find the indices of each value in the unique values array
    value_indices = np.searchsorted(unique_values, data)
    noisy_data = data.astype(float) + value_to_noise_array[value_indices].astype(float)
    return noisy_data


def add_decimal_noise(decimal_column, noise_level):
    """
    Adds random noise to decimal data values.

    Parameters:
    ----------
    decimal_column : list or numpy array
        A list or array of decimal (floating-point) values to which noise will be added.
    noise_level : float
        The standard deviation of the noise distribution.

    Returns:
    -------
    numpy array
        An array of noisy decimal values.

    """
    # Convert the input column to a NumPy array and flatten it
    data = np.array(decimal_column).flatten()

    # Identify unique values in the data
    unique_values = np.unique(data)

    # Map each unique value to a noise value
    try:
        value_to_noise = {
            value: make_number_noise(noise_level) for value in unique_values
        }
    except Exception as e:
        st.error(f"Error in generating noise values: {e}")
        return np.array([])  # Return an empty array in case of error

    # Create an array of noise values corresponding to the unique values
    value_to_noise_array = np.array([value_to_noise[val] for val in unique_values])

    # Find the indices of each value in the unique values array
    value_indices = np.searchsorted(unique_values, data)

    # Add noise to the original data
    noisy_data = data.astype(float) + value_to_noise_array[value_indices].astype(float)

    return noisy_data


def hash_value(value):
    """
    Hashes a given value using SHA-256 and returns a shortened hash.

    Parameters:
    ----------
    value : any type
        The value to be hashed. It will be converted to a string before hashing.

    Returns:
    -------
    str
        A shortened SHA-256 hash of the input value, consisting of the first 20 characters.

    """
    # Convert the value to a string
    value_str = str(value)
    # Generate the SHA-256 hash
    sha256_hash = hashlib.sha256(value_str.encode("utf-8")).hexdigest()
    # Shorten the hash to the first 20 characters
    short_sha256_hash = sha256_hash[:20]
    return short_sha256_hash


def add_noise_to_dates(dates):
    """
    Adds random noise to a list of dates by shifting them forward or backward.

    Parameters:
    ----------
    dates : list of pyodbc.Row
        A list of pyodbc.Row objects where each row contains a date in its first column.

    Returns:
    -------
    list
        A list of dates with random noise added, with the noise being a number of days shifted.

    """
    # Extract date values from pyodbc.Row objects
    date_list = [row[0] for row in dates]

    # Convert the list to a pandas Series
    dates_series = pd.Series(date_list)

    # Create a dictionary to store noise for each unique date
    unique_dates = dates_series.unique()
    noise_dict = {date: np.random.randint(-6, 4) for date in unique_dates}

    # Apply noise to the column
    noisy_dates = dates_series.apply(lambda x: x + pd.Timedelta(days=noise_dict[x]))

    # Convert the Series back to a list
    return noisy_dates.tolist()


def encode_data(data):
    """
    Encodes string data by applying SHA-256 hashing to each unique value.
    Parameters:
    ----------
    data : list
        A list of data values to be encoded.
        The function assumes that the data consists of strings or can be converted to strings.
    Returns:
    -------
    np.ndarray
        An array of encoded values, where each unique value in the input data is hashed.
    """
    # Encode string data  using SHA-256 hashing
    data_str = np.array([str(value) for value in data])

    # Get the unique values
    unique_vals = np.unique(data_str)

    # Create a dictionary to store the encoding for each unique value
    encoding_dict = {val: hash_value(val) for val in unique_vals}

    # Encode the data using the dictionary
    encoded_data = np.array([encoding_dict[val] for val in data_str])

    return encoded_data


def check_cpu_memory_usage(start_time, initial_memory):
    """
    Monitors and reports the performance of a code segment,
    including execution time and memory usage.

    """

    end_time = time.time()
    final_memory = psutil.virtual_memory().percent

    execution_time = end_time - start_time
    memory_usage = final_memory - initial_memory

    performance = [
        {"Metric": "Execution Time (seconds)", "Value": execution_time},
        {"Metric": "Memory Usage (%)", "Value": memory_usage},
    ]

    # Display performance metrics using Streamlit
    st.header("Algorithm Performance:")
    st.table(performance)


def get_tables_column_is_joined_to(table, column, join_columns_mapping):
    """
    Retrieves all tables that are joined to the specified column in the given table.

    Parameters:
    ----------
    table : str
        The name of the table containing the column of interest.
    column : str
        The name of the column in the specified table.
    join_columns_mapping : list of dict
        A list of dictionaries where each dictionary represents a join condition.
        Each dictionary contains:
        - "left_col": The column on the left side of the join (format: 'table.column').
        - "right_col": The column on the right side of the join (format: 'table.column').

    Returns:
    -------
    list of str
        A list of table names that are joined to the specified column, including the original table.

    Example:
    -------
    join_columns_mapping = [
        {"left_col": "employees.department_id", "right_col": "departments.id"},
        {"left_col": "departments.manager_id", "right_col": "managers.id"}
    ]
    result = get_tables_column_is_joined_to("employees", "department_id", join_columns_mapping)
    # result would be ['employees', 'departments']
    """

    joined_tables = [table]  # Start with the original table

    for join in join_columns_mapping:
        left_table, left_col = join["left_col"].split(".")
        right_table, right_col = join["right_col"].split(".")

        # Check if the column is on the left side of the join
        if join["left_col"] == f"{table}.{column}":
            if right_table.strip() not in joined_tables:
                joined_tables.append(right_table.strip())

        # Check if the column is on the right side of the join
        elif join["right_col"] == f"{table}.{column}":
            if left_table.strip() not in joined_tables:
                joined_tables.append(left_table.strip())

    return joined_tables


def process_columns(
    conn, valid_table_columns_mapping, modified_columns, join_columns_mapping
):
    """
    Processes and updates columns in the database based on the modifications specified.

    This function iterates through each table and column,
    creating new columns as needed and applying anonymization techniques.
    It also handles columns involved in joins by creating corresponding columns
    in joined tables.

    Parameters:
    ----------
    cursor : pyodbc.Cursor
        A cursor object to execute SQL queries on the SQL Server database.
    conn : pyodbc.Connection
        A connection object to the SQL Server database.
    valid_table_columns_mapping : dict
        A dictionary mapping table names to their columns, which are valid for modification.
    modified_columns : dict
        A dictionary mapping original column names to their new names after modification.
    join_columns_mapping : list
        A list of dictionaries representing join conditions,
        where each dictionary has 'left_col' and
        'right_col' specifying the columns involved in the join.

    Returns:
    -------
    None
        This function does not return any value. It performs updates directly on the database.
    """
    cursor = conn.cursor()
    for table, columns in valid_table_columns_mapping.items():
        for original_col in columns:
            if original_col in modified_columns:

                new_column_name = modified_columns[original_col]
                column_base_name = new_column_name.split("_")[0]
                column_type, length, is_primary_key, is_foreign_key = get_column_type(
                    cursor, table, column_base_name
                )
                if join_columns_mapping:
                    # Get all the tables that this column is joined to
                    joined_tables = get_tables_column_is_joined_to(
                        table, original_col, join_columns_mapping
                    )
                    if joined_tables:
                        # Create new columns for all tables involved in the join
                        for join_table in joined_tables:
                            check_and_create_columns(
                                conn,
                                join_table,
                                modified_columns[original_col],
                                "nvarchar",
                                "64",
                            )
                            apply_anonymization_tech_to_column(
                                conn,
                                join_table,
                                column_base_name,
                                is_primary_key,
                                is_foreign_key,
                                column_type,
                            )
                elif (
                    is_primary_key
                    or is_foreign_key
                    or column_type in ["nvarchar", "char", "varchar", "text", "ntext"]
                ):
                    check_and_create_columns(
                        conn, table, modified_columns[original_col], "nvarchar", "64"
                    )
                    apply_anonymization_tech_to_column(
                        conn,
                        table,
                        column_base_name,
                        is_primary_key,
                        is_foreign_key,
                        column_type,
                    )
                else:
                    check_and_create_columns(
                        conn, table, new_column_name, column_type, length
                    )
                    apply_anonymization_tech_to_column(
                        conn,
                        table,
                        column_base_name,
                        is_primary_key,
                        is_foreign_key,
                        column_type,
                    )


def compare_penguin_original_data_with_noisy_data():
    """
    Compares the original and noisy data for the 'culmen_length_mm' column in the Penguin dataset.

    This function loads the Penguin dataset, applies noise to the 'culmen_length_mm' column,
    and visualizes the original and noisy data using Kernel Density Estimation (KDE) plots.

    Steps performed:
    1. Load the dataset from a CSV file.
    2. Convert the 'culmen_length_mm' column to numeric and drop any NaN values.
    3. Apply noise to the numeric column.
    4. Create KDE plots for both the original and noisy data.
    5. Display the plots side by side in a Streamlit app.

    """

    st.header("Evaluation of Noise Technique with Penguin Dataset")

    # Load the dataset
    file_path = "pensize_clean.csv"  # Replace with your file's path
    df = pd.read_csv(file_path)

    # Convert column to numeric and drop NaN values
    decimal_column = pd.to_numeric(df["culmen_length_mm"], errors="coerce")
    decimal_column = decimal_column.dropna()

    # Original data
    original_data = np.array(decimal_column).flatten()

    # Apply noise to the data
    noisy_data = add_decimal_noise(original_data, 5)
    df["culmen_length_mm_noisy"] = pd.Series(noisy_data)

    # Plot original data
    g1 = sns.FacetGrid(df, hue="species", height=5)
    g1.map(sns.kdeplot, "culmen_length_mm", shade=True).add_legend(title="Original")

    # Plot noisy data
    g2 = sns.FacetGrid(df, hue="species", height=5)
    g2.map(sns.kdeplot, "culmen_length_mm_noisy", shade=True).add_legend(
        title="With Noise"
    )

    # Create two columns in Streamlit
    col1, col2 = st.columns(2)

    # Render the charts side by side
    with col1:
        fig1 = g1.figure
        st.pyplot(fig1)

    with col2:
        fig2 = g2.figure
        st.pyplot(fig2)

    st.write(
        "Conclusion:\n- In the original data, the species are more clearly separated."
        " With the addition of noise, the groups tend to mix together more."
    )


def main():
    """
    The main function that control the algorithm.
    """
    modified_columns = []
    st.session_state.query = ""
    st.session_state.selected_indices_input = ""
    st.title("Data Anonymization Algorithm")

    # Text area for SQL query input
    sql_query = st.text_area(
        "Enter your SQL query:",
        value=st.session_state.query,
        key="sql_query_area",
        height=200,
    )

    # Update session state if query changes
    if sql_query and sql_query != st.session_state.query:
        st.session_state.query = sql_query

    if st.session_state.query:
        if not sql_query:
            st.error("Please enter a SQL query.")
            return

        try:
            # Database connection
            conn = pyodbc.connect(
                "DRIVER={ODBC Driver 17 for SQL Server};"
                "SERVER=localhost\\MSSQLSERVER02;"
                "DATABASE=test500;"
                "Trusted_Connection=yes;"
                "Connection Timeout=60;"
                "Command Timeout=60;"
            )
            cursor = conn.cursor()

            # Extract tables and columns from the query
            results = extract_tables_and_columns(sql_query)
            table_columns_mapping, tables_columns_list, join_columns_mapping = results

            # Validate columns in the tables
            valid_table_columns_mapping = validate_columns_in_tables(
                cursor, table_columns_mapping
            )

            # Collect all column names
            all_columns = [
                column
                for columns in valid_table_columns_mapping.values()
                for column in columns
            ]

            if all_columns:  # Check if there are columns available for selection
                # Display available columns
                display_all_columns(all_columns)

                # Text input for selecting columns to modify
                selected_indices_input = st.text_input(
                    "Enter the numbers of the columns you want to modify, separated by commas:",
                    value=st.session_state.selected_indices_input,
                    key="selected_indices_area",
                )

                start_time = time.time()
                initial_memory = psutil.virtual_memory().percent

                if selected_indices_input:
                    if (
                        selected_indices_input
                        != st.session_state.selected_indices_input
                    ):
                        st.session_state.selected_indices_input = selected_indices_input
                        selected_columns = select_columns_to_modify(
                            all_columns, selected_indices_input
                        )

                        # Modify column names
                        modified_columns = modify_column_names(selected_columns)

                        # Process columns: create new columns and apply anonymization
                        process_columns(
                            conn,
                            valid_table_columns_mapping,
                            modified_columns,
                            join_columns_mapping,
                        )

            else:
                st.error("No columns available for modification.")

            # Rebuild the SQL query with modified columns
            if selected_indices_input:
                new_sql_query = add_modified_columns_to_sql_query(
                    sql_query, modified_columns, "_noisy"
                )
                (
                    modified_table_columns_mapping,
                    modified_tables_columns_list,
                    modified_join_columns_mapping,
                ) = extract_tables_and_columns(new_sql_query)

                # Display original and modified SQL queries
                st.text_area("Original SQL Query:", sql_query, height=200)
                st.text_area("Modified SQL Query:", new_sql_query, height=200)

                # Display data from the original query
                st.header("Original Data:")
                df_original = pd.read_sql(sql_query, conn)
                df_original.columns = tables_columns_list
                st.table(df_original.head(5))

                # Display data from the modified query
                st.header("Modified Data:")
                df_modified = pd.read_sql(new_sql_query, conn)
                df_modified.columns = modified_tables_columns_list
                st.table(df_modified.head(5))

                # Optional: Compare original and noisy data
                compare_penguin_original_data_with_noisy_data()

                # Close the database connection
                cursor.close()
                conn.close()
                # Optional: Check CPU and memory usage
                check_cpu_memory_usage(start_time, initial_memory)

        except Exception as e:
            st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
