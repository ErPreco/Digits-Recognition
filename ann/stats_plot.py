import matplotlib.pyplot as plt

def fine_tuning_plot(data: dict[str, dict[str | float, list[float]]], data_kind: str, tuned_param_name: str, tuned_params: list[str | float], legend_format: str=None) -> None:
    """
    Initializes a figure with the plots of "data" (training and test sets) of a tuned parameter.

    Args:
        data (dict[str, dict[str | float, list[float]]]): The training and test data.
        data_kind (str): The kind of data (typically 'loss' or 'accuracy').
        tuned_param_name (str): The name of the tuned paramter.
        tuned_params (list[str | float]): The values of the tuned parameter.
        legend_format (str): The print format of the values in the legend.
    """
    plt.figure()
    for param in tuned_params:
        color = None
        for mode in ['train', 'test']:
            line, = plt.plot(data[mode][param])
            formatted_legend = (f'{param}' if legend_format is None else f'{param:{legend_format}}') + f' {mode}'   # avoids None format error e.g. 'int'
            line.set_label(formatted_legend)
            if mode == 'train':
                color = line.get_color()   # keeps the same color for the train and test line
            else:
                line.set_linestyle('--')
                line.set_color(color)
    plt.title((f'{tuned_param_name} {data_kind}').capitalize())
    plt.xlabel('epochs')
    plt.ylabel(data_kind)
    plt.legend(loc=('upper right' if data_kind.lower() == 'loss' else 'lower right'))

def model_plot(data: dict[str, list[float]], data_kind: str, model_params: str=None) -> None:
    """Initializes a figure with the plots of "data" (training and test sets) of the model.

    Args:
        data (dict[str, list[float]]): The training and test data.
        data_kind (str): The kind of data (typically 'loss' or 'accuracy').
        model_params (str): The model parameters to display as subtitle.
    """
    plt.figure(figsize=(7, 6.3) if model_params is not None else None)   # enlarge the figure if the model parameters are given
    color = None
    for mode in ['train', 'test']:
        line, = plt.plot(data[mode], label=mode)
        if mode == 'train':
            color = line.get_color()   # keeps the same color for the train and test line
        else:
            line.set_linestyle('--')
            line.set_color(color)
    plt.suptitle(f'MLP {data_kind}')
    if model_params is not None:
        plt.title(model_params, fontsize=10)
    plt.xlabel('epochs')
    plt.ylabel(data_kind)
    plt.legend(loc=('upper right' if data_kind.lower() == 'loss' else 'lower right'))

def show() -> None:
    """Shows the plots initialized in fine_tuning_plot module.
    """
    plt.show()
