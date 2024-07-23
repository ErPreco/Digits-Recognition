######################################################################################################################
# Installing torch, torchvision and torchaudio with CUDA extension is required in order to use NVIDIA GPU cuda cores #
# CUDA 12.1 is recommended from https://pytorch.org/                                                                 #
######################################################################################################################

import ast
import os
import torch
import torch.nn.functional as F

from ann import MLP
from ann import stats_plot as splt
from ann import tmnist_data_loader as dl
from ann import trainer

import ui
from ui.menu import Menu
from ui.select import SelectWindow
from ui.test import TestWindow
from ui.test import Pixel
from ui.train import TrainWindow

SCREEN_RES: dict[str, int] = {'width': 28, 'height': 28}
N_CLASSES: int = 10

script_path = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH: str = os.path.join(script_path, 'ann\\models')
MODELS_FILE_NAME: str = 'models.txt'
MODELS_STATE_PATH: str = os.path.join(script_path, 'ann\\models\\states')
MODELS_STATE_FILE_EXTENSION: str = '.pth.tar'

model: MLP = None
epochs: int = None

def set_device() -> torch.device:
    """Returns the device (GPU or CPU).
    """
    if torch.cuda.is_available():
        print('GPU enabled!')
        device = torch.device('cuda:0')
    else:
        print('You are not using the GPU')
        device = torch.device('cpu')
    return device

device: torch.device = set_device()

# Fine tuning
#
# dl.download_dataset()
# dls = dl.load_batches(shuffle_data=True)
# inputs_dim = SCREEN_RES['width'] * SCREEN_RES['height']
#
# activation function:
# trainer.activation_tuning(inputs_dim, [2048, 1024, 128, 10], 3e-4, 'uniform', 0.0, dls, device, 50)                             # ReLU
# weights and biases initial distribution:
# trainer.init_kind_tuning(inputs_dim, [2048, 1024, 128, 10], 'relu', 3e-4, 0.0, dls, device, 50)                                 # Normal
# learning rate:
# trainer.lr_tuning(inputs_dim, [2048, 1024, 128, 10], 'relu', 'normal', 0.0, dls, device, 50)                                    # [1e-4, 1e-5]
# trainer.lr_tuning(inputs_dim, [2048, 1024, 128, 10], 'relu', 'normal', 0.0, dls, device, 50, [1e-4, 8e-5, 5e-5, 2e-5, 1e-5])    # 8e-5
# dropout rate:
# trainer.dor_tuning(inputs_dim, [2048, 1024, 128, 10], 'relu', 8e-5, 'normal', dls, device, 50)                                  # [0.6, 0.8]
# trainer.dor_tuning(inputs_dim, [2048, 1024, 128, 10], 'relu', 8e-5, 'normal', dls, device, 50, [0.6, 0.65, 0.7, 0.75, 0.8])     # 0.75
# epochs:
# trainer.epoch_tuning(inputs_dim, [2048, 1024, 128, 10], 'relu', 8e-5, 'normal', 0.75, dls, device)                              # 50

def train(params: dict[str, list[int] | str | float | int], show_stats: bool, show_graph: bool, file_name: str=None) -> None:
    """Creates a MLP model with the given hyper parameters, trains it and save the model if "file_name" is not None.

    Args:
        params (dict[str, list[int]  |  str  |  float  |  int]): The list of hyper paramters.
        show_stats (bool): Whether show the stats per epoch or not.
        show_graph (bool): Whether show the loss and accuracy graphs or not.
        file_name (str): The name of the file in which save the model.
    """
    dl.download_dataset()
    dls = dl.load_batches(shuffle_data=True)

    inputs_dim = SCREEN_RES['width'] * SCREEN_RES['height']
    params['layers'].append(10)
    model = MLP(inputs_dim, params['layers'], params['activation'], params['lr'], params['init_kind'], params['dor']).to(device)
    loss, accuracy = trainer.train(model, dls, MLP.accuracy, device, params['epochs'], show_stats)   # actually trains the model

    print(f'\nFinal results with the following MLP model over {params['epochs']} epochs:')
    params_str = f'in_dim: {inputs_dim}, layers: {params['layers']}, act_fn: {params['activation']}, lr: {params['lr']:.0e}, init_kind: {params['init_kind']}, dor: {params['dor']}'
    print(params_str)
    print(f'train : loss={loss['train'][-1]:.4f}, acc={accuracy['train'][-1]:.4f}')
    print(f'test  : loss={loss['test'][-1]:.4f}, acc={accuracy['test'][-1]:.4f}')

    if show_graph:
        splt.model_plot(loss, 'loss', params_str)
        splt.model_plot(accuracy, 'accuracy', params_str)
        splt.show()
    
    if file_name is not None:
        print('\nModel saving...')
        
        # adds "(copy)" to the file name if another one exists
        while os.path.exists(f'{MODELS_STATE_PATH}\\{file_name}{MODELS_STATE_FILE_EXTENSION}'):
            file_name += ' (copy)'
        file = f'{MODELS_STATE_PATH}\\{file_name}{MODELS_STATE_FILE_EXTENSION}'
        torch.save(model, file)   # saves the model state

        # saves the model hyper paramters
        with open(f'{MODELS_PATH}\\{MODELS_FILE_NAME}', 'a') as f:
            f.write(f'{file_name}\t\t:\t{params}\n')
        print(f'Model saved: "{file}"')

def select_model(stats_path: str) -> None:
    """Creates an MLP model by the given file.

    Args:
        stats_path (str): The path of the model stats file.
    """
    print('\nModel selecting...')
    with open(f'{MODELS_PATH}\\{MODELS_FILE_NAME}', 'r') as f:
        model_name = os.path.basename(stats_path).split('.')[0]
        params = None
        # keeps the model hyper parameters as a dictionary
        for line in f:
            split_line = line.split(':', 1)
            saved_model_name = split_line[0].strip()

            if saved_model_name == model_name:
                saved_model_params = split_line[1].strip()
                params = ast.literal_eval(saved_model_params)
                break
        
        if params is not None:
            global model
            model = MLP(SCREEN_RES['width'] * SCREEN_RES['height'], params['layers'], params['activation'],
                        params['lr'], params['init_kind'], params['dor']).to(device)
            global epochs
            epochs = params['epochs']
            print(f'Selected model: {model_name}')
            print(', '.join(f"{key}: {value}" for key, value in params.items()))

            print('\nModel stats loading...')
            model = torch.load(stats_path)   # loads the model stats
            print(f'Model stats loaded from "{stats_path}"')
        else:
            print(f'No model found named "{model_name}" in file "{f.name}"')

def predict(grid: list[list[Pixel]], window: TestWindow) -> None:
    """Predicts the number drew on the given grid.

    Args:
        grid (list[list[Pixel]]): The grid of pixels.
    """
    input = torch.tensor([int(px.get_value()) for row in grid for px in row])   # puts the rows of pixels in one line ([row1, row2, ...])
    input = input.to(device).to(torch.float32)   # matches the weights type

    logits = model(input)
    preds = F.softmax(logits, -1)
    
    window.update_prediction(preds.tolist())

def train_opt() -> None:
    """Performs the training window.
    """
    window = TrainWindow(SCREEN_RES['width'], N_CLASSES, train)
    ui.exec(window.get_window())

def select_opt() -> None:
    """Performs the select window.
    """
    window = SelectWindow(select_model)
    ui.exec(window.get_window())

def test_opt() -> None:
    """Perfroms the test window.
    """
    window = TestWindow(N_CLASSES, SCREEN_RES['width'], SCREEN_RES['height'], predict)
    ui.exec(window.get_window())

if __name__ == '__main__':
    menu = Menu(train_opt, select_opt, test_opt)
    ui.exec(menu.get_window())
