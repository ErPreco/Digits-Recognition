#################################################################################################################
# Installing kaggle is required in order to download datasets from API                                          #
# Create .kaggle folder in ~/user and create a new token from https://www.kaggle.com user settings if necessary #
#################################################################################################################

import csv
import kaggle
import os
import random
import torch
import torch.utils.data as tud

from log import progression

_KAGGLE_DATASET: str = 'nimishmagre/tmnist-typeface-mnist'
_script_path = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH: str = os.path.join(_script_path, 'dataset')
_file: str = ''

def download_dataset() -> None:
    """Downloads the dataset.
    """
    print()
    # downloads the dataset whether it is not done yet
    if os.listdir(_DATASET_PATH) == []:
        kaggle.api.authenticate()
        print('Kaggle API authenticated!')
        
        print('Downloading dataset...')
        if not os.path.exists(_DATASET_PATH):
            os.mkdir(_DATASET_PATH)
        kaggle.api.dataset_download_files(_KAGGLE_DATASET, path=_DATASET_PATH, unzip=True)
        print('Dataset downloaded!')
    else:
        print('Dataset already downloaded')
    global _file
    if _file == '':
        _file = _DATASET_PATH + '/' + os.listdir(_DATASET_PATH)[0]

def load_data() -> list[list[int]]:
    """Returns the targets and the features into a single list [M,1+N] where M=num_samples and N=num_features

    Returns:
        list[list[int]]: The list of the entire dataset
    """
    ds = []
    with open(_file, 'r') as f:
        print('\nFetching data from the downloaded dataset:')
        imgs_data = csv.reader(f, delimiter=',')
        i, n_imgs = 0, sum(1 for _ in imgs_data) - 1
        f.seek(0)
        next(imgs_data)   # skips the header
        for row in imgs_data:
            ds.append([int(x) for x in row[1:]])   # keeps [target, pixels] and converts them into int
            i += 1
            progression.print_progress_bar(i, n_imgs)
    return ds

def load_batches(batch_size: int=128, train_perc: float=0.75, shuffle_data: bool=False) -> dict[str, tud.DataLoader]:
    """Returns the train and test DataLoader divided into batches.

    Args:
        batch_size (int): The size of each batch. Defaults to 256.
        train_perc (float): The percentage of data that are used for training. Defaults to 0.8.
        shuffle_data (bool): Whether shuffle the data before batches creation or not. Defaults to False.

    Returns:
        dict[str, DataLoader]: The dictionary of the train and test batches. ['train'] returns the train DataLoader, ['test'] the test one.
    """
    data = load_data()
    print('\nBatches preparation...')
    if shuffle_data:
        random.shuffle(data)
    # splits the data into (train_perc)% train and (1 - train_perc)% test
    bound_idx = int(len(data) * train_perc)
    ds_train_x = torch.tensor([x[1:] for x in data[:bound_idx]])    # features train dataset
    ds_train_y = torch.tensor([x[0] for x in data[:bound_idx]])     # targets train dataset
    ds_test_x = torch.tensor([x[1:] for x in data[bound_idx:]])     # features test dataset
    ds_test_y = torch.tensor([x[0] for x in data[bound_idx:]])      # targets test dataset

    # completes train and test datasets ([features, target] for each sample)
    # [[[f1', f2', ...], t'], [[f1'', f2'', ...], t''], ...]
    ds_train = tud.TensorDataset(ds_train_x, ds_train_y)    # len(ds_train) == len(data) * train_perc
    ds_test = tud.TensorDataset(ds_test_x, ds_test_y)       # len(ds_test) == len(data) * (1 - train_perc)
    
    # creates train and test batches
    # batch_i = [[[f1', f2', ...], [f1'', f2'', ...], ...], [t', t'', ...]]
    # [batch_1, batch_2, ...]
    dl_train = tud.DataLoader(ds_train, batch_size=batch_size, shuffle=True)    # len(dl_train) == len(ds_train) / batch_size
    dl_test = tud.DataLoader(ds_test, batch_size=batch_size, shuffle=False)     # len(dl_test) == len(ds_test) / batch_size
    print('Batches completed!')
    return {'train': dl_train, 'test': dl_test}
