import tkinter as tk
from typing import Callable

from ui.window import Window

class Menu(Window):
    """
    The menu window from which train, select or test a MLP model.

    Attributes
    ----------

    Private:
    
        _train_btn (Button)
            The button for training a model
        _select_btn (Button)
            The button for selecting a saved model
        _test_btn (Button)
            The button for testing the selected model
    

    Methods
    -------

    Private:

        _open_window
            Opens the window corresponding to its button
    """

    def __init__(self, train_fn: Callable[[None], None], select_fn: Callable[[None], None], test_fn: Callable[[None], None],
                 res: str=None, title: str='Digits recognition menu', zoomed: bool=True) -> None:
        super().__init__(res, title, zoomed)

        frame = tk.Frame(self.window)
        frame.pack(expand=True)

        self._train_btn: tk.Button = tk.Button(frame, text='TRAIN', font=('Arial', 20), width=10, height=1, command=lambda: self._open_window(train_fn))
        self._train_btn.grid(row=0, column=0, padx=15)

        self._select_btn: tk.Button = tk.Button(frame, text='SELECT', font=('Arial', 20), width=10, height=1, command=lambda: self._open_window(select_fn))
        self._select_btn.grid(row=0, column=1, padx=15)

        self._test_btn: tk.Button = tk.Button(frame, text='TEST', font=('Arial', 20), width=10, height=1, command=lambda: self._open_window(test_fn))
        self._test_btn.grid(row=0, column=2, padx=15)
    
    def _open_window(self, fn: Callable[[None], None]) -> None:
        """Open the window corresponding to the given function.

        Args:
            fn (Callable[[None], None]): The function to perform the new window logic.
        """
        self._train_btn.config(state=tk.DISABLED)
        self._select_btn.config(state=tk.DISABLED)
        self._test_btn.config(state=tk.DISABLED)
        fn()
        self._train_btn.config(state=tk.ACTIVE)
        self._select_btn.config(state=tk.ACTIVE)
        self._test_btn.config(state=tk.ACTIVE)

