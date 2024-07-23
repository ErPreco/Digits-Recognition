import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud
from typing import Callable

from ann import MLP
from ann import stats_plot as splt
from log import progression

def train(model: MLP, dls: dict[str, tud.DataLoader], acc_fn: Callable[[torch.tensor, torch.tensor], float],
          device: torch.device, epochs: int, print_stats: bool=False) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Trains the given MLP model.

    Args:
        model (MLP): The MLP model.
        dls (dict[str, DataLoader]): The dictionary of the train and test dataset.
        acc_fn (Callable[[tensor, tensor], float]): The function that calculates the accuracy.
        device (device): The device on which to preform processing.
        epochs (int): The number of epochs.
        print_stats (bool): Whether print the loss and accuracy per epoch or not.

    Returns:
        tuple[dict[str, list[float]], dict[str, list[float]]]: The tuple of the loss and accuracy per epoch ({'train': [], 'test': []}).
    """
    print('\nMLP model training:')

    if not print_stats:
        i_tot = 0
        progression.print_progress_bar(i_tot, epochs * (len(dls['train']) + len(dls['test'])))

    def _forward(dl: tud.DataLoader, mode: str) -> tuple[float, float]:
        """Trains the given MLP model over the entire batches of the given mode.

        Args:
            dl (DataLoader): The batches of the dataset.
            mode (str): The current mode to train ('train' or 'test')

        Returns:
            tuple[float, float]: The mean of the loss and the accuracy.
        """
        if mode == 'train':
            model.train()
        else:
            model.eval()
        
        running_loss, running_acc = [], []
        if print_stats:
            progression.print_progress_bar(0, len(dl), prefix=f'{mode.capitalize():5s}', print_end='\t')
        for i, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)

            if mode == 'train':
                model.optimizer.zero_grad()
            
            x = x.to(torch.float32)   # matches the weights type
            logits = model(x)   # forward propagation
            loss_batch = model.loss(logits, y)
            acc = acc_fn(F.softmax(logits, -1), y)
            if mode == 'train':
                loss_batch.backward()   # backward propagation
                model.optimizer.step()   # gradient descent

            running_loss.append(loss_batch.item())
            running_acc.append(acc)

            if print_stats:
                # updates the progress bar for each epoch
                progression.print_progress_bar(i + 1, len(dl), prefix=f'{mode.capitalize():5s}', print_end='\t')
            else:
                # updates the progress bar for the entire training
                nonlocal i_tot
                i_tot += 1
                progression.print_progress_bar(i_tot, epochs * (len(dls['train']) + len(dls['test'])))
        
        return np.mean(running_loss), np.mean(running_acc)
    
    loss = {'train': [], 'test': []}
    accuracy = {'train': [], 'test': []}

    for epoch in range(epochs):
        if print_stats:
            print(f'Epoch {epoch + 1:03d}/{epochs:03d}')

        for mode in ['train', 'test']:
            dl = dls[mode]
            loss_epoch, acc_epoch = _forward(dl, mode)
            loss[mode].append(loss_epoch)
            accuracy[mode].append(acc_epoch)
            if print_stats:
                print(f'[loss={loss_epoch:.4f}, acc={acc_epoch:.4f}]')
    
    return loss, accuracy

def _init_fine_tuning_cache() -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Initializes the dictionary for caching loss and accuracy of a parameter tuning.

    Returns:
        tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]: The tuple of (losses, accuracies)
    """
    return {'train': {}, 'test': {}}, {'train': {}, 'test': {}}

def activation_tuning(input_dim: int, layers: list[int], lr: float, init_kind: str, dor: float, dls: dict[str, tud.DataLoader],
                      device: torch.device, epochs: int, acts: list[str]=['relu', 'linear', 'sigmoid', 'tanh']) -> None:
    """Performs activation function tuning between ReLU, Linear, Sigmoid and Tanh.

    Args:
        input_dim (int): The dimension of the input layer.
        layers (list[int]): The dimensions of the layers (the last must be 10 due to classes).
        lr (float): The learning rate.
        init_kind (str): The weights and biases initial distribution ('uniform', 'zeros', 'normal', 'xavier').
        dor (float): The dropout rate.
        dls (dict[str, DataLoader]): The dictionary of the train and test dataset.
        device (device): The device on which to preform processing.
        epochs (int): The number of epochs.
        acts (list[str]): The activation functions to test.
    """
    losses, accuracies = _init_fine_tuning_cache()
    
    for act in acts:
        print(f'\nProcessing fine tuning on "{act}" activation function:', end='')
        net = MLP(input_dim, layers, act, lr, init_kind, dor).to(device)
        loss, accuracy = train(net, dls, MLP.accuracy, device, epochs)
        for mode in ['train', 'test']:
            losses[mode][act] = loss[mode]
            accuracies[mode][act] = accuracy[mode]
    print()
    
    splt.fine_tuning_plot(losses, 'loss', 'activation function', acts)
    splt.fine_tuning_plot(accuracies, 'accuracy', 'activation function', acts)
    splt.show()

def init_kind_tuning(input_dim: int, layers: list[int], activation: str, lr: float, dor: float, dls: dict[str, tud.DataLoader],
                     device: torch.device, epochs: int, init_kinds: list[str]=['uniform', 'zeros', 'normal', 'xavier']) -> None:
    """Performs weights and biases initialization tuning between Uniform, Zeros, Normal and Xavier.

    Args:
        input_dim (int): The dimension of the input layer.
        layers (list[int]): The dimensions of the layers (the last must be 10 due to classes).
        activation (str): The kind of the activation function ('sigmoid', 'relu', 'linear', 'tanh').
        lr (float): The learning rate.
        dor (float): The dropout rate.
        dls (dict[str, DataLoader]): The dictionary of the train and test dataset.
        device (device): The device on which to preform processing.
        epochs (int): The number of epochs.
        init_kinds (list[str]): The weights and biases initial distributions to test.
    """
    losses, accuracies = _init_fine_tuning_cache()

    for init_kind in init_kinds:
        print(f'\nProcessing fine tuning on "{init_kind}" initialization kind for weights and biases:', end='')
        net = MLP(input_dim, layers, activation, lr, init_kind, dor).to(device)
        loss, accuracy = train(net, dls, MLP.accuracy, device, epochs)
        for mode in ['train', 'test']:
            losses[mode][init_kind] = loss[mode]
            accuracies[mode][init_kind] = accuracy[mode]
    print()

    splt.fine_tuning_plot(losses, 'loss', 'initial params distribution', init_kinds)
    splt.fine_tuning_plot(accuracies, 'accuracy', 'initial params distribution', init_kinds)
    splt.show()

def lr_tuning(input_dim: int, layers: list[int], activation: str, init_kind: str, dor: float, dls: dict[str, tud.DataLoader],
              device: torch.device, epochs: int, lrs: list[float]=[1e-2, 1e-3, 1e-4, 1e-5]) -> None:
    """Performs learning rate tuning between 1e-2, 1e-3, 1e-4 and 1e-5 by default.

    Args:
        input_dim (int): The dimension of the input layer.
        layers (list[int]): The dimensions of the layers (the last must be 10 due to classes).
        activation (str): The kind of the activation function ('sigmoid', 'relu', 'linear', 'tanh').
        init_kind (str): The kind of the weights and biases distribution ('uniform', 'zeros', 'normal', 'xavier').
        dor (float): The dropout rate.
        dls (dict[str, DataLoader]): The dictionary of the train and test dataset.
        device (device): The device on which to preform processing.
        epochs (int): The number of epochs.
        lrs (list[float]): The learning rates to test.
    """
    losses, accuracies = _init_fine_tuning_cache()

    for lr in lrs:
        print(f'\nProcessing fine tuning on {lr:.0e} learning rate:', end='')
        net = MLP(input_dim, layers, activation, lr, init_kind, dor).to(device)
        loss, accuracy = train(net, dls, MLP.accuracy, device, epochs)
        for mode in ['train', 'test']:
            losses[mode][lr] = loss[mode]
            accuracies[mode][lr] = accuracy[mode]
    print()

    splt.fine_tuning_plot(losses, 'loss', 'learning rate', lrs, '.0e')
    splt.fine_tuning_plot(accuracies, 'accuracy', 'learning rate', lrs, '.0e')
    splt.show()

def dor_tuning(input_dim: int, layers: list[int], activation: str, lr: float, init_kind: str, dls: dict[str, tud.DataLoader],
              device: torch.device, epochs: int, dors: list[float]=[0.0, 0.2, 0.4, 0.6, 0.8]) -> None:
    """Performs dropout rate tuning between 0.0, 0.2, 0.4, 0.6 and 0.8 by default.

    Args:
        input_dim (int): The dimension of the input layer.
        layers (list[int]): The dimensions of the layers (the last must be 10 due to classes).
        activation (str): The kind of the activation function ('sigmoid', 'relu', 'linear', 'tanh').
        lr (float): The learning rate.
        init_kind (str): The kind of the weights and biases distribution ('uniform', 'zeros', 'normal', 'xavier').
        dls (dict[str, DataLoader]): The dictionary of the train and test dataset.
        device (device): The device on which to preform processing.
        epochs (int): The number of epochs.
        dors (list[float]): The dropout rates to test.
    """
    losses, accuracies = _init_fine_tuning_cache()

    for dor in dors:
        print(f'\nProcessing fine tuning on {dor} dropout rate:', end='')
        net = MLP(input_dim, layers, activation, lr, init_kind, dor).to(device)
        loss, accuracy = train(net, dls, MLP.accuracy, device, epochs)
        for mode in ['train', 'test']:
            losses[mode][dor] = loss[mode]
            accuracies[mode][dor] = accuracy[mode]
    print()

    splt.fine_tuning_plot(losses, 'loss', 'dropout rate', dors)
    splt.fine_tuning_plot(accuracies, 'accuracy', 'dropout rate', dors)
    splt.show()

def epoch_tuning(input_dim: int, layers: list[int], activation: str, lr: float, init_kind: str, dor: float, dls: dict[str, tud.DataLoader],
                 device: torch.device, epochs_list: list[int]=[30, 50, 80, 100]) -> None:
    """Performs dropout rate tuning between 30, 50, 80 and 100 by default.

    Args:
        input_dim (int): The dimension of the input layer.
        layers (list[int]): The dimensions of the layers (the last must be 10 due to classes).
        activation (str): The kind of the activation function ('sigmoid', 'relu', 'linear', 'tanh').
        lr (float): The learning rate.
        init_kind (str): The kind of the weights and biases distribution ('uniform', 'zeros', 'normal', 'xavier').
        dor (float): The dropout rate.
        dls (dict[str, DataLoader]): The dictionary of the train and test dataset.
        device (device): The device on which to preform processing.
        epochs_list (list[int]): The epochs to test.
    """
    losses, accuracies = _init_fine_tuning_cache()

    for epochs in epochs_list:
        print(f'\nProcessing fine tuning on {epochs} epochs:', end='')
        net = MLP(input_dim, layers, activation, lr, init_kind, dor).to(device)
        loss, accuracy = train(net, dls, MLP.accuracy, device, epochs)
        for mode in ['train', 'test']:
            losses[mode][epochs] = loss[mode]
            accuracies[mode][epochs] = accuracy[mode]
    print()

    splt.fine_tuning_plot(losses, 'loss', 'epochs', epochs_list)
    splt.fine_tuning_plot(accuracies, 'accuracy', 'epochs', epochs_list)
    splt.show()
