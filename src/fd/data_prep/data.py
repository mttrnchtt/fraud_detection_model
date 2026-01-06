import pandas as pd


def load_raw_csv(path: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame with minimal validation.

    Args:
        path: The file path to the CSV file.

    Returns:
        A pandas.DataFrame containing the data from the CSV file.

    Raises:
        ValueError: If the CSV file is missing the required 'Class' or 'Time' columns.
    """
    df = pd.read_csv(path)

    # minimal sanity checks for this project
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in the CSV.")
    if "Time" not in df.columns:
        raise ValueError("Expected a 'Time' column in the CSV.")

    return df


def chronological_split(df: pd.DataFrame, val_size: float, test_size: float, time_col: str = "Time") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time order: first -> train, middle -> val, last -> test.
    Sizes are fractions of the whole dataset (e.g., 0.15).
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError("Train size became non-positive; check val_size/test_size.")

    df_train = df_sorted.iloc[:n_train].reset_index(drop=True)
    df_val = df_sorted.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df_sorted.iloc[n_train + n_val:].reset_index(drop=True)

    return df_train, df_val, df_test
