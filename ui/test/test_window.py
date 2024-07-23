from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import tkinter as tk
from typing import Callable

from ui.window import Window
from ui.test import Canvas
from ui.test import Pixel

class TestWindow(Window):
    """
    The window for testing a model by drawing.

    Attributes
    ----------

    Private:

        _n_classes (int)
            The number of classes
        _canvas (Canvas)
            The canvas on which drawing
        _data (list[float])
            The confidence values of the classes
        _ax (Axes)
            The axes of the confidences bar plots
        _plot (FigureCanvasTkAgg)
            The confidences bar plot
        _pred_l (Label)
            The label of the prediction
    
    
    Methods
    -------

    Private:

        _on_clean_click
            Cleans the canvas, the bar plot and sets the prediction to "None"
    
    Public:

        update_prediction
            Updates the confidences bar plot and the prediction label
    """

    def __init__(self, n_classes: int, n_px_width: int, n_px_height: int, predict_fn: Callable[[list[list[Pixel], 'TestWindow']], None], px_size: int=24, res: str=None, title: str='Digits recognition test window') -> None:
        super().__init__(res, title)

        frame = tk.Frame(self.window)
        frame.pack(expand=True)

        self._n_classes: int = n_classes

        self._canvas: Canvas = Canvas(frame, n_px_width * px_size, n_px_height * px_size, px_size)
        self._canvas.get_canvas().grid(row=0, column=0, rowspan=3, padx=20, pady=20)

        label_font = 'Arial'
        label_font_size = 20

        # buttons to clear the canvas and to predict the number
        buttons_frame = tk.Frame(frame)
        buttons_frame.grid(row=0, column=1, padx=20, pady=10)

        clean_btn = tk.Button(buttons_frame, text='CLEAN', font=(label_font, label_font_size), width=10, command=self._on_clean_click)
        clean_btn.grid(row=0, column=0, padx=15)

        predict_btn = tk.Button(buttons_frame, text='PREDICT', font=(label_font, label_font_size), width=10, command=lambda: predict_fn(self._canvas.get_grid(), self))
        predict_btn.grid(row=0, column=1, padx=15)

        # bar plot of confidences
        figure = Figure(dpi=100)
        self._ax: Axes = figure.add_subplot(111)
        
        self._data: list[float] = {}
        for i in range(n_classes):
            self._data[str(i)] = 0.0
        
        self._ax.bar(list(self._data.keys()), list(self._data.values()))
        self._ax.set_ylim(0, 1)
        
        self._plot: FigureCanvasTkAgg = FigureCanvasTkAgg(figure, frame)
        self._plot.draw()
        self._plot.get_tk_widget().grid(row=1, column=1, padx=20, pady=10)

        # prediction label
        self._pred_l: tk.Label = tk.Label(frame, text='Prediction: None', font=(label_font, label_font_size))
        self._pred_l.grid(row=2, column=1, padx=20, pady=10)
    
    def _on_clean_click(self) -> None:
        """Cleans the canvas, the bar plot and sets the prediction to "None".
        """
        self._canvas.clean()
        self.update_prediction(None)

    def update_prediction(self, preds: list[float]) -> None:
        """Updates the confidences bar plot and the prediction label.

        Args:
            preds (list[float]): The list of the confidences.
        """
        for i in range(self._n_classes):
            self._data[str(i)] = preds[i] if preds is not None else 0
        
        self._ax.clear()
        self._ax.set_ylim(0, 1)
        self._ax.bar(list(self._data.keys()), list(self._data.values()))
        self._plot.draw()

        if preds is not None:
            max_value = max(preds)
            self._pred_l.config(text=f'Prediction: {preds.index(max_value)}')
        else:
            self._pred_l.config(text='Prediction: None')
