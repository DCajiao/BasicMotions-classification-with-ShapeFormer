import os
import json
import torch
import numpy as np
import logging
import zipfile
import requests
from datetime import datetime
from torch.utils.data import Dataset

from Dataset import load_segment_data, load_UEA_data

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def Setup(args):
    """
    Initializes and prepares the experiment directory structure.

    Creates output subdirectories for model checkpoints, predictions, and TensorBoard summaries.
    Also saves a configuration file (`configuration.json`) for reproducibility.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: A configuration dictionary including derived paths and parameters.
    """
    config = args.__dict__  # configuration dictionary
    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    output_dir = os.path.join(output_dir, config['data_path'], initial_timestamp.strftime("%Y-%m-%d_%H-%M"))
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}' as a configuration.json".format(output_dir))

    return config


def create_dirs(dirs):
    """
    Creates a list of directories if they do not already exist.

    Args:
        dirs (list of str): List of directory paths to create.

    Returns:
        int: 0 if directories created successfully, exits with error otherwise.
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def Initialization(config):
    """
    Initializes training environment by setting seeds and selecting device.

    Sets the random seed if specified, and determines whether to use CPU or GPU.

    Args:
        config (dict): Configuration dictionary with keys like 'seed' and 'gpu'.

    Returns:
        torch.device: The selected computation device (CPU or CUDA).
    """

    if config['seed'] is not None:
        torch.manual_seed(config['seed'])
    device = torch.device('cuda' if (torch.cuda.is_available() and config['gpu'] != '-1') else 'cpu')
    logger.info("Using device: {}".format(device))
    if device == 'cuda':
        logger.info("Device index: {}".format(torch.cuda.current_device()))
    return device


def Data_Loader(config):
    """
    Loads dataset according to the specified configuration.

    Automatically selects the appropriate loading function depending on whether the
    dataset is segmented (e.g., HAR) or UEA-style (.ts format).

    Args:
        config (dict): Configuration containing 'data_dir'.

    Returns:
        tuple: A dataset object (features, labels, metadata, etc.) ready for use.
    """
    if config['data_dir'].split('/')[1] == 'Segmentation':
        Data = load_segment_data.load(config)  # Load HAR (WISDM V2) and Ford Datasets
    else:
        Data = load_UEA_data.load(config)  # Load UEA *.ts Data
    return Data


def Data_Verifier(config):
    """
    Verifies whether the dataset is already downloaded and unzipped.

    Creates the data directory if missing. For UEA datasets, downloads and extracts
    data from an external archive if not found locally.

    Args:
        config (dict): Configuration dictionary containing 'data_path'.
    """

    if not os.path.exists(config['data_path']):
        os.makedirs(os.path.join(os.getcwd(), config['data_path']))
    directories = [name for name in os.listdir(config['data_path']) if os.path.isdir(os.path.join(config['data_path'], name))]

    if directories:
        logger.info(f"The {config['data_path'].split('/')[-2]} data is already existed")
    else:
        if config['data_path'].split('/')[1] == 'UEA':
            file_url = 'http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip'
            Downloader(file_url, 'UEA')

    if config['data_path'].split('/')[-2] == 'UEA':
        config['data_path'] = os.path.join(config['data_path'], 'Multivariate_ts')


def Downloader(file_url, problem):
    """
    Downloads and extracts a dataset zip archive from a given URL.

    Tracks and logs the download progress, extracts its contents,
    and removes the original ZIP file after extraction.

    Args:
        file_url (str): URL to download the ZIP file from.
        problem (str): Subdirectory name under 'Dataset/' where data will be saved.
    """

    # Define the path to download
    path_to_download = os.path.join('Dataset/', problem)
    # Send a GET request to download the file
    response = requests.get(file_url, stream=True)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the downloaded file
        file_path = os.path.join(path_to_download, 'Multivariate2018_ts.zip')
        with open(file_path, 'wb') as file:
            # Track the progress of the download
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 * 100  # 1KB
            downloaded_size = 0

            for data in response.iter_content(block_size):
                file.write(data)
                downloaded_size += len(data)

                # Calculate the download progress percentage
                progress = (downloaded_size / total_size) * 100

                # Print the progress message
                logger.info(f' Download in progress: {progress:.2f}%')

        # Extract the contents of the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(path_to_download)

        # Remove the downloaded zip file
        os.remove(file_path)

        logger.info(f'{problem} Datasets downloaded and extracted successfully.')
        
    else:
        logger.error(f'Failed to download the {problem} please update the file_url')
    return


class dataset_class(Dataset):
    """
    Custom PyTorch Dataset for loading time series data and labels.

    Wraps the input features and labels as tensors and allows indexing
    for use in DataLoader pipelines.

    Args:
        data (np.ndarray): Time series features (num_samples x num_dims x num_timesteps).
        label (np.ndarray): Corresponding class labels (num_samples,).

    Methods:
        __getitem__(ind): Returns a tuple (data_tensor, label_tensor, index).
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):

        x = self.feature[ind]
        x = x.astype(np.float32)

        y = self.labels[ind]  # (num_labels,) array

        data = torch.tensor(x)
        label = torch.tensor(y)

        return data, label, ind

    def __len__(self):
        return len(self.labels)



