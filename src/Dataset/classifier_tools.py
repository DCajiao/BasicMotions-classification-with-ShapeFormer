import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

__author__ = "Chang Wei Tan"


# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc

def prepare_inputs_deep_learning(train_inputs, test_inputs, window_len=40, stride=20,
                                 val_size=1, random_state=1234, verbose=1):
    """
    Prepare multivariate time series data for deep learning models by extracting sliding window subsequences.

    The function splits the training data into training and validation series, extracts fixed-length 
    overlapping subsequences from the original time series, and assigns the corresponding labels 
    using majority voting within each window. It outputs data shaped as (n_samples, window_len, n_features) 
    which is suitable for most neural network architectures.

    Parameters:
        train_inputs (object): An object with `data` and `label` attributes for the training set.
        test_inputs (object): An object with `data` and `label` attributes for the test set.
        window_len (int): Length of the sliding window for subsequence extraction. Default is 40.
        stride (int): Step size between consecutive windows. Default is 20.
        val_size (int): Number of time series to reserve for validation. Default is 1.
        random_state (int): Seed for reproducibility in train/validation split. Default is 1234.
        verbose (int): Verbosity level for logging. Default is 1.

    Returns:
        tuple:
            - X_train (np.ndarray): Extracted training subsequences of shape (n_train_samples, window_len, n_features).
            - y_train (np.ndarray): Corresponding training labels.
            - X_val (np.ndarray or None): Extracted validation subsequences or None if not used.
            - y_val (np.ndarray or None): Corresponding validation labels or None if not used.
            - X_test (np.ndarray): Extracted test subsequences.
            - y_test (np.ndarray): Corresponding test labels.
    """

    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    if len(train_inputs) > val_size:
        train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                    test_size=val_size,
                                                    random_state=random_state)
    else:
        train_series = range(len(train_inputs))
        val_series = None

    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if val_series is None:
        X_val = None
        y_val = None
    else:
        X_val = []
        y_val = []
        for i in val_series:
            this_series = train_inputs.data[i]
            this_series_labels = train_inputs.label[i]
            subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                           window_size=window_len,
                                                           stride=stride)
            [X_val.append(x) for x in subsequences]
            [y_val.append(x) for x in sub_label]
        X_val = np.array(X_val)
        y_val = np.array(y_val)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # r, m, n = X_train.shape
    # X_train = np.column_stack((np.repeat(np.arange(m), n), X_train.reshape(m * n, -1)))
    # X_train_df = pd.DataFrame(X_train)
    # y_train = pd.Series(y_train, dtype="category")
    return X_train, y_train, X_val, y_val, X_test, y_test


def extract_subsequences(X_data, y_data, window_size=30, stride=1, norm=False):
    """
    Extract fixed-length overlapping subsequences from a multivariate time series.

    For each subsequence window, the label is determined by the majority class (mode) 
    among the labels within that window. Optionally applies z-normalization.

    Parameters:
        X_data (np.ndarray): Input time series of shape (timesteps, features).
        y_data (np.ndarray): Sequence of labels per timestep of shape (timesteps,).
        window_size (int): Length of each subsequence window. Default is 30.
        stride (int): Step size between consecutive windows. Default is 1.
        norm (bool): Whether to apply z-normalization to each subsequence. Default is False.

    Returns:
        tuple:
            - subsequences (np.ndarray): Array of shape (n_subsequences, window_size, features).
            - labels (np.ndarray): Array of shape (n_subsequences,) with majority labels per window.
    """

    data_len, data_dim = X_data.shape
    subsequences = []
    labels = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end > data_len:
            break
        tmp = X_data[i:end, :]
        if norm:
            # usually z-normalisation is required for TSC
            scaler = StandardScaler()
            tmp = scaler.fit_transform(tmp)
        subsequences.append(tmp)
        label = stats.mode(y_data[i:end]).mode[0]
        labels.append(label)
        count += 1
    return np.array(subsequences), np.array(labels)
