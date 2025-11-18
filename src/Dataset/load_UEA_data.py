import os
import numpy as np
import logging
import random
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sktime.utils.load_data import load_from_tsfile_to_dataframe
# from sktime.datasets._data_io import load_from_tsfile_to_dataframe
from sktime.datasets import load_from_tsfile_to_dataframe

logger = logging.getLogger(__name__)


def load(config):
    """
    Load and preprocess multivariate time series data from the UEA archive format (.ts files).

    If preprocessed `.npy` data exists, it is loaded. Otherwise, the function reads UEA .ts format files,
    encodes labels, handles padding for variable-length time series, optionally normalizes the data,
    and splits the dataset into training, validation, and test sets.

    Parameters:
        config (dict): Dictionary with keys:
            - 'data_dir': str, path to the dataset folder.
            - 'val_ratio': float, fraction of training data for validation.
            - 'Norm': bool, whether to normalize the dataset.

    Returns:
        dict: A dictionary containing:
            - 'train_data', 'train_label'
            - 'val_data', 'val_label'
            - 'test_data', 'test_label'
            - 'All_train_data', 'All_train_label'
            - 'max_len': int, maximum sequence length
    """
    # Build data
    Data = {}
    problem = config['data_dir'].split('/')[-1]

    if os.path.exists(config['data_dir'] + '/' + problem + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' +
                           problem + '.npy', allow_pickle=True)

        Data['max_len'] = Data_npy.item().get('max_len')
        Data['All_train_data'] = Data_npy.item().get('All_train_data')
        Data['All_train_label'] = Data_npy.item().get('All_train_label')
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['val_data'] = Data_npy.item().get('val_data')
        Data['val_label'] = Data_npy.item().get('val_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(
            len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(
            len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(
            len(Data['test_label'])))

    else:
        logger.info("Loading and preprocessing data ...")
        train_file = config['data_dir'] + "/" + problem + "_TRAIN.ts"
        test_file = config['data_dir'] + "/" + problem + "_TEST.ts"
        train_df, y_train = load_from_tsfile_to_dataframe(train_file)
        test_df, y_test = load_from_tsfile_to_dataframe(test_file)

        y_train = LabelEncoder().fit_transform(y_train)
        y_test = LabelEncoder().fit_transform(y_test)

        train_lengths = train_df.applymap(lambda x: len(x)).values
        test_lengths = test_df.applymap(lambda x: len(x)).values
        train_max_seq_len = int(np.max(train_lengths[:, 0]))
        test_max_seq_len = int(np.max(test_lengths[:, 0]))
        max_seq_len = np.max([train_max_seq_len, test_max_seq_len])

        X_train = process_ts_data(train_df, max_seq_len, normalise=False)
        X_test = process_ts_data(test_df, max_seq_len, normalise=False)

        if config['Norm']:
            mean, std = mean_std(X_train)
            mean = np.repeat(mean, max_seq_len).reshape(
                X_train.shape[1], max_seq_len)
            std = np.repeat(std, max_seq_len).reshape(
                X_train.shape[1], max_seq_len)
            X_train = mean_std_transform(X_train, mean, std)
            X_test = mean_std_transform(X_test, mean, std)

        Data['max_len'] = max_seq_len
        Data['All_train_data'] = X_train
        Data['All_train_label'] = y_train

        if config['val_ratio'] > 0:
            train_data, train_label, val_data, val_label = split_dataset(
                X_train, y_train, config['val_ratio'])
        else:
            val_data, val_label = [None, None]

        logger.info(
            "{} samples will be used for training".format(len(train_label)))
        logger.info(
            "{} samples will be used for validation".format(len(val_label)))
        logger.info("{} samples will be used for testing".format(len(y_test)))

        Data['train_data'] = train_data
        Data['train_label'] = train_label
        Data['val_data'] = val_data
        Data['val_label'] = val_label
        Data['test_data'] = X_test
        Data['test_label'] = y_test

        np.save(config['data_dir'] + "/" + problem, Data, allow_pickle=True)

    return Data


def split_dataset(data, label, validation_ratio):
    """
    Stratified train/validation split for time series classification.

    Ensures that the label distribution is preserved in both sets.

    Parameters:
        data (np.ndarray): Full dataset array of shape (n_samples, n_channels, seq_len).
        label (np.ndarray): Array of integer-encoded labels.
        validation_ratio (float): Proportion of data to use for validation.

    Returns:
        tuple: (train_data, train_label, val_data, val_label)
    """

    splitter = model_selection.StratifiedShuffleSplit(
        n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(
        *splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label


def fill_missing(x: np.array, max_len: int, vary_len: str = "suffix-noise", normalise: bool = True):
    """
    Fill missing or shorter subsequences in time series with appropriate padding or scaling.

    Depending on `vary_len`, uses different strategies for padding:
        - "zero": replace NaNs with zeros.
        - "prefix-suffix-noise": pad both ends with small random noise.
        - "uniform-scaling": stretch original sequence uniformly.
        - default: replace NaNs with random noise.

    Parameters:
        x (np.ndarray): Input 2D array of shape (n_instances, sequence_length) with possible NaNs.
        max_len (int): Target length for sequences.
        vary_len (str): Padding method. One of ["zero", "prefix-suffix-noise", "uniform-scaling"].
        normalise (bool): Whether to apply z-normalization after filling.

    Returns:
        np.ndarray: Array with completed sequences of shape (n_instances, max_len).
    """
    if vary_len == "zero":
        if normalise:
            x = StandardScaler().fit_transform(x)
        x = np.nan_to_num(x)
    elif vary_len == 'prefix-suffix-noise':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            diff_len = int(0.5 * (max_len - seq_len))

            for j in range(diff_len):
                x[i, j] = random.random() / 1000

            for j in range(diff_len, seq_len):
                x[i, j] = series[j - seq_len]

            for j in range(seq_len, max_len):
                x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    elif vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)

            for j in range(max_len):
                scaling_factor = int(j * seq_len / max_len)
                x[i, j] = series[scaling_factor]
            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    else:
        for i in range(len(x)):
            for j in range(len(x[i])):
                if np.isnan(x[i, j]):
                    x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]

    return x


def process_ts_data(x, max_len, vary_len: str = "suffix-noise", normalise: bool = False):
    """
    Convert sktime-style pandas DataFrame of multivariate time series into a 3D NumPy array.

    Each time series is reshaped into fixed-length format and padded using `fill_missing`.

    Parameters:
        x (pd.DataFrame): Input DataFrame with shape (n_instances, n_dimensions), where each cell contains a pd.Series.
        max_len (int): Target fixed sequence length.
        vary_len (str): Padding strategy for variable-length sequences.
        normalise (bool): Whether to normalize each sequence.

    Returns:
        np.ndarray: 3D array of shape (n_instances, n_dimensions, max_len)
    """

    num_instances, num_dim = x.shape
    columns = x.columns
    # max_len = np.max([len(X[columns[0]][i]) for i in range(num_instances)])
    output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)
    for i in range(num_dim):
        for j in range(num_instances):
            lengths = len(x[columns[i]][j].values)
            end = min(lengths, max_len)
            output[j, i, :end] = x[columns[i]][j].values
        output[:, i, :] = fill_missing(
            output[:, i, :], max_len, vary_len, normalise)
    return output


def mean_std(train_data):
    """
    Compute mean and standard deviation across time steps for normalization.

    Parameters:
        train_data (np.ndarray): Time series data of shape (n_samples, n_channels, seq_len).

    Returns:
        tuple:
            - mean (np.ndarray): Mean per channel.
            - std (np.ndarray): Max standard deviation per channel.
    """

    m_len = np.mean(train_data, axis=2)
    mean = np.mean(m_len, axis=0)

    s_len = np.std(train_data, axis=2)
    std = np.max(s_len, axis=0)

    return mean, std


def mean_std_transform(train_data, mean, std):
    """
    Apply normalization using precomputed mean and standard deviation.

    Parameters:
        train_data (np.ndarray): Input data of shape (n_samples, n_channels, seq_len).
        mean (np.ndarray): Mean per channel.
        std (np.ndarray): Std per channel.

    Returns:
        np.ndarray: Normalized data.
    """

    return (train_data - mean) / std
