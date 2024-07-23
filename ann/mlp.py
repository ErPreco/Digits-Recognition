import torch
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    """
    An MLP torch model.

    Attributes
    ----------

    Private:

        _input_dim (int)
            The dimension of the input layer
        _layers (list[int])
            The dimensions of the layers (the last must be 10 due to classes)
        _activation (str)
            The kind of the activation function ('sigmoid', 'relu', 'linear', 'tanh')
        _lr (float)
            The learning rate
        _init_kind (str)
            The weights and biases initial distribution ('uniform', 'zeros', 'normal', 'xavier')
        _dor (float)
            The dropout rate

    Public:

        loss (CrossEntropyLoss)
            The Cross Entropy loss function
        optimizer (Adam)
            The optimizer of the model
    
    
    Methods
    -------

    Private:

        _get_activation
            Selects the activation function
        _get_layers
            Creates the layers with the given dimensions
        _init_weights_and_biases
            Initializes weights and biases according to the given init_kind (uniform, zeros, normal, xavier)
        _compile
            Defines the loss function and the optimizer
    
    Public:

        accuracy (static)
            Calculates the accuracy by the given prediction and targets
        forward
            The forward method over the layers
    """

    def __init__(self, input_dim: int, layers: list[int]=[10], activation: str='sigmoid', lr: float=8e-3, init_kind: str='uniform', dor: float=0.0):
        super(MLP, self).__init__()
        self._input_dim: int = input_dim
        self._lr: float = lr
        self._dor: nn.Dropout = nn.Dropout(dor)
        self._activation: (nn.ReLU | nn.Identity | nn.Sigmoid | nn.Tanh) = self._get_activation(activation)
        self._layers: nn.Sequential = self._gen_layers(layers)
        self._init_kind: str = init_kind
        self._init_weights_and_biases(init_kind)
        self._compile()
    
    @staticmethod
    def accuracy(preds: torch.tensor, targets: torch.tensor) -> float:
        """Calculates the accuracy by the given predictions and targets.

        Args:
            preds (tensor): The reliability of each class for each sample.
            targets (tensor): The ground truth for each sample.

        Returns:
            float: The accuracy over the given predictions and targets.
        """
        _, preds = torch.max(preds, -1)   # gets the indexes of the maximum value, which correspond to the class prediction
        return float(torch.where(targets == preds, 1, 0).sum() / len(preds))

    def _get_activation(self, act_kind: str) -> (nn.ReLU | nn.Identity | nn.Sigmoid | nn.Tanh):
        """Returns the acrivation function of the kind given.

        Args:
            act_kind (str): The kind of the function ('relu', 'linear', 'sigmoid', 'tanh').

        Raises:
            Exception: function kind not supported.

        Returns:
            (ReLU | Identity | Sigmoid | Tanh): The class representing the function.
        """
        if act_kind == 'relu':
            return nn.ReLU()
        elif (act_kind is None) or (act_kind == 'linear'):
            return nn.Identity()
        elif act_kind == 'sigmoid':
            return nn.Sigmoid()
        elif act_kind == 'tanh':
            return nn.Tanh()
        else:
            raise Exception(f'Error! Activation "{act_kind}" is not supported.')

    def _gen_layers(self, layers_size: list[int]) -> nn.Sequential:
        """Returns the sequential of the layers with the given dimensions but the input layer.
        Pre-initializing the activation function and the dropout rate is required.

        Args:
            layers_size (list[int]): The dimensions of the layers.

        Returns:
            Sequential: The sequential of the layers.
        """
        layers = []
        input_dim = self._input_dim
        for layer_idx, layer_size in enumerate(layers_size):
            layers.append(nn.Linear(input_dim, layer_size))
            if layer_idx < len(layers_size) - 1:
                layers.append(self._activation)
                layers.append(self._dor)
            input_dim = layer_size
        return nn.Sequential(*layers)
    
    def _init_weights_and_biases(self, init_kind: str) -> None:
        """Initializes the weights and the biases according to the given kind of the distribution.

        Args:
            init_kind (str): The kind of the distribution ('uniform', 'zeros', 'normal', 'xavier').
        """
        for layer in self._layers:
            # does not initialize the activation function and the dropout rate layers
            if not isinstance(layer, nn.Linear):
                continue

            if init_kind == 'zeros':
                nn.init.zeros_(layer.weight)
            elif init_kind == 'uniform':
                nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
            elif init_kind == 'normal':
                nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            elif init_kind == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.00)
    
    def _compile(self) -> None:
        """Initializes the loss function as the Cross Entropy and the parameters optimizer.
        """
        self.loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.optimizer: optim.Adam = optim.Adam(self.parameters(), lr=self._lr)
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        """The forward method over the layers.

        Args:
            x (tensor): The input.

        Returns:
            tensor: The output of the sequential.
        """
        return self._layers(x)
