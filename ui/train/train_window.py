import tkinter as tk
from typing import Callable

from ui.window import Window

class TrainWindow(Window):
    """
    The window for training a MLP model by setting its hyper parameters.

    Attributes
    ----------

    Private:
    
        _train_fn (Callable[[dict[str, list[int] | str | float | int], bool, bool, str], None])
            The training function
        _layers_v (Entry)
            The entry for the layers size
        _activation (StringVar)
            The string for the activation function dropdown menu
        _lr_v (Entry)
            The entry for the learning rate
        _init_kind (StringVar)
            The string for the weights and biases initial distribution dropdown menu
        _dor_v (Scale)
            The slider for the dropout rate
        _epochs_v (Entry)
            The entry for the epochs
        _show_stats (BooleanVar)
            The boolean for the checkbutton that allows showing the stats per epoch
        _show_graphs (BooleanVar)
            The boolean for the checkbutton that allows showing the loss and accuracy graphs
        _save_model (BooleanVar)
            The boolean for the checkbutton that allows saving the trained model
        _file_name_v (Entry)
            The entry of the file name where saving the trained model
        _train_btn (Button)
            The button that triggers the train function
    
    
    Methods
    -------

    Private:
    
        _get_model_params
            Returns a dictionary of the hyper parameters inserted via UI
        _on_train_click
            The method invoked by the button to train the model
    """

    def __init__(self, screen_width: int, n_classes: int, train_fn: Callable[[dict[str, list[int] | str | float | int], bool, bool, str], None], res: str=None, title: str='Digits recognition train window') -> None:
        super().__init__(res, title)

        self._train_fn: Callable[[dict[str, list[int] | str | float | int], bool, bool, str], None] = train_fn

        label_font = 'Arial'
        label_font_size = 15
        label_padx = 5
        label_pady = 5

        # container frame
        frame = tk.Frame(self.window)
        frame.pack(expand=True)

        # hyper params frame
        params_frame = tk.Frame(frame)
        params_frame.pack(padx=40, pady=15)

        # inputs info
        input_l = tk.Label(params_frame, text='Inputs', font=(label_font, label_font_size))
        input_l.grid(row=0, column=0, padx=label_padx, pady=label_pady)
        input_v = tk.Label(params_frame, text=f'{screen_width * screen_width} ({screen_width}x{screen_width})', font=(label_font, label_font_size))
        input_v.grid(row=0, column=1, padx=label_padx, pady=label_pady)

        # classes info
        classes_l = tk.Label(params_frame, text='Classes', font=(label_font, label_font_size))
        classes_l.grid(row=1, column=0, padx=label_padx, pady=label_pady)
        classes_v = tk.Label(params_frame, text=f'{n_classes}', font=(label_font, label_font_size))
        classes_v.grid(row=1, column=1, padx=label_padx, pady=label_pady)

        # layers size entry
        layers_l = tk.Label(params_frame, text='Layers size\n(excluding classes)', font=(label_font, label_font_size))
        layers_l.grid(row=2, column=0, padx=label_padx, pady=label_pady)
        self._layers_v: tk.Entry = tk.Entry(params_frame, foreground='gray', font=(label_font, label_font_size))
        layers_placeholder = 'e.g. 1000, 500, 20'
        self._layers_v.insert(0, layers_placeholder)
        self._layers_v.bind('<FocusIn>', lambda event: self.on_entry_click(self._layers_v, layers_placeholder))
        self._layers_v.bind('<Return>', lambda event: self.on_focus_out(self._layers_v, layers_placeholder))
        self._layers_v.bind('<Escape>', lambda event: self.on_focus_out(self._layers_v, layers_placeholder))
        self._layers_v.grid(row=2, column=1, padx=label_padx, pady=label_pady)

        # activation function dropdown menu
        act_fn_l = tk.Label(params_frame, text='Activation function', font=(label_font, label_font_size))
        act_fn_l.grid(row=3, column=0, padx=label_padx, pady=label_pady)
        act_fn_options = ['ReLU', 'Linear', 'Sigmoid', 'Tanh']
        self._activation: tk.StringVar = tk.StringVar(params_frame, act_fn_options[0])
        act_fn_v: tk.OptionMenu = tk.OptionMenu(params_frame, self._activation, *act_fn_options)
        act_fn_v.config(font=(label_font, label_font_size), width=8)
        act_fn_v.grid(row=3, column=1, padx=label_padx, pady=label_pady)

        # learning rate entry
        lr_l = tk.Label(params_frame, text='Learning rate', font=(label_font, label_font_size))
        lr_l.grid(row=4, column=0, padx=label_padx, pady=label_pady)
        self._lr_v: tk.Entry = tk.Entry(params_frame, foreground='gray', font=(label_font, label_font_size))
        lr_placeholder = 'e.g. 0.001'
        self._lr_v.insert(0, lr_placeholder)
        self._lr_v.bind('<FocusIn>', lambda event: self.on_entry_click(self._lr_v, lr_placeholder))
        self._lr_v.bind('<Return>', lambda event: self.on_focus_out(self._lr_v, lr_placeholder))
        self._lr_v.bind('<Escape>', lambda event: self.on_focus_out(self._lr_v, lr_placeholder))
        self._lr_v.grid(row=4, column=1, padx=label_padx, pady=label_pady)

        # weights and biases initial distribution dropdown menu
        init_kind_l = tk.Label(params_frame, text='Weights and biases\ninitial distribution', font=(label_font, label_font_size))
        init_kind_l.grid(row=5, column=0, padx=label_padx, pady=label_pady)
        init_kind_options = ['Uniform', 'Zeros', 'Normal', 'Xavier']
        self._init_kind: tk.StringVar = tk.StringVar(params_frame, init_kind_options[0])
        init_kind_v: tk.OptionMenu = tk.OptionMenu(params_frame, self._init_kind, *init_kind_options)
        init_kind_v.config(font=(label_font, label_font_size), width=8)
        init_kind_v.grid(row=5, column=1, padx=label_padx, pady=label_pady)

        # dropout rate slider
        dor_l = tk.Label(params_frame, text='Dropout rate', font=(label_font, label_font_size))
        dor_l.grid(row=6, column=0, padx=label_padx, pady=label_pady)
        self._dor_v: tk.Scale = tk.Scale(params_frame, from_=0, to=100, orient=tk.HORIZONTAL, length=200)
        self._dor_v.grid(row=6, column=1, padx=label_padx, pady=label_pady)

        # epochs entry
        epochs_l = tk.Label(params_frame, text='Epochs', font=(label_font, label_font_size))
        epochs_l.grid(row=7, column=0, padx=label_padx, pady=label_pady)
        self._epochs_v: tk.Entry = tk.Entry(params_frame, foreground='gray', font=(label_font, label_font_size))
        epochs_placeholder = 'e.g. 30'
        self._epochs_v.insert(0, epochs_placeholder)
        self._epochs_v.bind('<FocusIn>', lambda event: self.on_entry_click(self._epochs_v, epochs_placeholder))
        self._epochs_v.bind('<Return>', lambda event: self.on_focus_out(self._epochs_v, epochs_placeholder))
        self._epochs_v.bind('<Escape>', lambda event: self.on_focus_out(self._epochs_v, epochs_placeholder))
        self._epochs_v.grid(row=7, column=1, padx=label_padx, pady=label_pady)

        # epochs stats checkbox
        epochs_stats_l = tk.Label(params_frame, text='Show stats per epoch\non terminal', font=(label_font, label_font_size))
        epochs_stats_l.grid(row=8, column=0, padx=label_padx, pady=label_pady)
        self._show_stats: tk.BooleanVar = tk.BooleanVar(value=False)
        epochs_stats_v = tk.Checkbutton(params_frame, variable=self._show_stats,
                                        command=lambda: self._show_stats.set(not self._show_stats.get()))   # awful work around because the checkbutton does not update the variable
        epochs_stats_v.grid(row=8, column=1, padx=label_padx, pady=label_pady)

        # graphs checkbox
        graphs_l = tk.Label(params_frame, text='Show loss and accuracy\ngraphs for train and test', font=(label_font, label_font_size))
        graphs_l.grid(row=9, column=0, padx=label_padx, pady=label_pady)
        self._show_graphs: tk.BooleanVar = tk.BooleanVar(value=False)
        graphs_v = tk.Checkbutton(params_frame, variable=self._show_graphs,
                                  command=lambda: self._show_graphs.set(not self._show_graphs.get()))   # awful work around because the checkbutton does not update the variable
        graphs_v.grid(row=9, column=1, padx=label_padx, pady=label_pady)

        # model file name entry
        file_name_l = tk.Label(params_frame, text='File name (no extension)', font=(label_font, label_font_size))
        self._file_name_v: tk.Entry = tk.Entry(params_frame, foreground='gray', font=(label_font, label_font_size))
        file_name_placeholder = 'type here ...'
        self._file_name_v.insert(0, file_name_placeholder)
        self._file_name_v.bind('<FocusIn>', lambda event: self.on_entry_click(self._file_name_v, file_name_placeholder))
        self._file_name_v.bind('<Return>', lambda event: self.on_focus_out(self._file_name_v, file_name_placeholder))
        self._file_name_v.bind('<Escape>', lambda event: self.on_focus_out(self._file_name_v, file_name_placeholder))
        
        def _show_hide_file_name():
            """Shows or hides the file name entry based on the checkbutton.
            """
            self._save_model.set(not self._save_model.get())
            if self._save_model.get():
                file_name_l.grid(row=11, column=0, padx=label_padx)
                self._file_name_v.grid(row=11, column=1, padx=label_padx)
            else:
                file_name_l.grid_forget()
                self._file_name_v.grid_forget()
        
        # model save checkbox
        save_l = tk.Label(params_frame, text='Save the MLP model', font=(label_font, label_font_size))
        save_l.grid(row=10, column=0, padx=label_padx, pady=label_pady)
        self._save_model: tk.BooleanVar = tk.BooleanVar(value=False)
        save_v = tk.Checkbutton(params_frame, variable=self._save_model, command=_show_hide_file_name)
        save_v.grid(row=10, column=1, padx=label_padx, pady=label_pady)

        # train button
        self._train_btn = tk.Button(frame, text='TRAIN', font=(label_font, int(label_font_size * 1.3)), command=self._on_train_click)
        self._train_btn.pack(pady=20)
    
    def _get_model_params(self) -> dict[str, list[int] | str | float | int]:
        """Returns a dictionary of the hyper paramters inserted via UI.

        Returns:
            dict[str, list[int] | str | float | int]: The dictionary of the hyper paramters.
        """
        params = {}
        params['layers'] = [int(l.strip()) for l in self._layers_v.get().split(',')]
        params['activation'] = self._activation.get().strip().lower()
        params['lr'] = float(self._lr_v.get().strip())
        params['init_kind'] = self._init_kind.get().lower()
        params['dor'] = int(self._dor_v.get()) / 100
        params['epochs'] = int(self._epochs_v.get().strip())
        return params
    
    def _on_train_click(self) -> None:
        """Gets the parameters and call the train function. Disables the button in the meanwhile.
        """
        self._train_btn.config(state=tk.DISABLED)

        params = self._get_model_params()
        show_stats = self._show_stats.get()
        show_graphs = self._show_graphs.get()
        file_name = self._file_name_v.get() if self._save_model.get() else None
        self._train_fn(params, show_stats, show_graphs, file_name)
        
        self._train_btn.config(state=tk.ACTIVE)
