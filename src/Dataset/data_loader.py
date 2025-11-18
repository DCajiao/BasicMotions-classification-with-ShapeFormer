import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

__author__ = "Chang Wei Tan and Navid Foumani"


def load_segmentation_data(file_path, data_type, norm=True, verbose=1):
    """
    Load and preprocess a multivariate time series segmentation dataset from a CSV file.

    This function reads labeled time series data from a CSV file and organizes it into 
    structured format for deep learning models. It filters and relabels the dataset 
    depending on whether the data type is 'Clean' or not, distinguishing 'distract' and 'drive' labels.

    Parameters:
        file_path (str): Path to the CSV file.
        data_type (str): Type of the dataset ("Clean" or other).
        norm (bool): Whether to apply z-normalization to each time series. Default is True.
        verbose (int): Verbosity level for printing progress messages. Default is 1.

    Returns:
        pd.DataFrame: A DataFrame where each row contains a dictionary with:
                      - 'data': np.ndarray of shape (seq_len, num_features)
                      - 'label': np.ndarray of shape (seq_len,)
    """

    if verbose > 0:
        print("[Data_Loader] Loading data from {}".format(file_path))

    df = pd.read_csv(file_path)
    drive = [3, 11]
    if data_type == "Clean":
        # Drop other class data ------------------------------------------------------------------------------
        # "X", "EyesCLOSEDneutral", "EyesOPENneutral", "LateBoredomLap"
        Other_Class = [0, 1, 2, 12]
        df = df.drop(np.squeeze(np.where(np.isin(df['label'], Other_Class))))
        distract = [4, 5, 6, 7, 8, 9, 10, 13, 14, 15]
        # -----------------------------------------------------------------------------------------------------
    else:
        distract = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15]
    all_series = df.series.unique()
    data = []
    for series in all_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)

        this_series.label = this_series.label.replace(distract, 0)
        this_series.label = this_series.label.replace(drive, 1)

        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)
    return data


def load_activity(file_path, data_type, norm=True, verbose=1):
    """
    Load and split human activity recognition dataset into training and test sets.

    The dataset is expected to have six columns: series ID, label, timestamp, and 3-axis data.
    Labels are encoded numerically and time series data is split into training and test sets 
    by sampling unique series identifiers. Normalization is applied optionally.

    Parameters:
        file_path (str): Path to the CSV file.
        data_type (str): Type of dataset (not used, but included for interface consistency).
        norm (bool): Whether to apply z-normalization to each time series. Default is True.
        verbose (int): Verbosity level for printing progress messages. Default is 1.

    Returns:
        tuple:
            - train_data (pd.DataFrame): DataFrame with training samples (columns: 'data', 'label').
            - test_data (pd.DataFrame): DataFrame with test samples (columns: 'data', 'label').
    """

    column_names = ['series', 'label',
                    'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv(file_path, header=None, names=column_names, comment=';')
    df.dropna(axis=0, how='any', inplace=True)
    LE = LabelEncoder()
    df['label'] = LE.fit_transform(df['label'])
    all_series = df.series.unique()
    train_series, test_series = train_test_split(
        [x for x in range(len(all_series))], test_size=6, random_state=1)
    train_data = []
    print("[Data_Loader] Loading Train Data")
    for series in train_series:
        series = series + 1
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        train_data.append(pd.DataFrame({"data": [series_data],
                                        "label": [series_labels]}, index=[0]))
    train_data = pd.concat(train_data)
    train_data.reset_index(drop=True, inplace=True)
    test_data = []
    print("[Data_Loader] Loading Test Data")
    for series in test_series:
        series = series + 1
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        test_data.append(pd.DataFrame({"data": [series_data],
                                       "label": [series_labels]}, index=[0]))
    test_data = pd.concat(test_data)
    test_data.reset_index(drop=True, inplace=True)
    return train_data, test_data


def load_ford_data(file_path, data_type, norm=True, verbose=1):
    """
    Load and preprocess Ford dataset for time series classification or segmentation.

    Each series in the dataset is processed individually. The function reshapes each 
    into a format compatible with deep learning: a multivariate time series and 
    associated per-timestamp labels. Labels and features are extracted from the data,
    with optional normalization.

    Parameters:
        file_path (str): Path to the CSV file.
        data_type (str): Type of the dataset (not used, but kept for consistency).
        norm (bool): Whether to apply z-normalization to each time series. Default is True.
        verbose (int): Verbosity level for printing progress messages. Default is 1.

    Returns:
        pd.DataFrame: A DataFrame where each row contains:
                      - 'data': np.ndarray of shape (seq_len, num_features)
                      - 'label': np.ndarray of shape (seq_len,)
    """

    if verbose > 0:
        print("[Data_Loader] Loading data from {}".format(file_path))
    df = pd.read_csv(file_path)
    all_series = df.series.unique()
    data = []

    for series in all_series:
        if verbose > 0:
            print("[Data_Loader] Processing series {}".format(series))
        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)

    return data
