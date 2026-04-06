import pandas as pd


def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def preprocess_data(df):
    """Preprocess the data by handling missing values and encoding categorical features."""
    # Example preprocessing steps
    df.fillna(df.mean(), inplace=True)  # Handle missing values
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding for categorical features
    return df
