import tkinter as tk
from tkinter import filedialog
from typing import Callable

from ui.window import Window

class SelectWindow(Window):
    """
    The window for the selection of a saved model.

    Attributes
    ----------

    Private:

        _select_fn (Callable[[str], None])
            The function that fetches the data from the file
        _selected_file_l (Label)
            The label that displays the file path
        _select_btn (Button)
            The button that triggers the selection function
    
    
    Methods
    -------

    Private:

        _open_file_dialog
            Opens the dialog for the file selection
        _select_model
            Triggers the selection function and closes the window
    """

    def __init__(self, select_fn: Callable[[str], None], res: str=None, title: str='Digits recognition select window') -> None:
        super().__init__(res, title)
        self.window.wm_minsize(width=300, height=0)

        self._select_fn: Callable[[str], None] = select_fn

        label_font = 'Arial'
        label_font_size = 15

        # container frame
        frame = tk.Frame(self.window)
        frame.pack(expand=True)

        # open file frame
        open_file_frame = tk.Frame(frame)
        open_file_frame.pack()

        # file selection
        self._selected_file_l: tk.Label = tk.Label(open_file_frame, text='...', font=(label_font, label_font_size))
        self._selected_file_l.grid(row=0, column=0, padx=5, pady=5)
        select_btn: tk.Button = tk.Button(open_file_frame, text='open', font=(label_font, label_font_size), command=self._open_file_dialog)
        select_btn.grid(row=0, column=1, padx=5, pady=5)

        # select button
        self._select_btn: tk.Button = tk.Button(frame, text='SELECT', font=(label_font, int(label_font_size * 1.3)), command=self._select_model)
        self._select_btn.pack(pady=20)
    
    def _open_file_dialog(self) -> None:
        """Opens the dialog for the file selection and saves the path into the "_selected_file_l" label.
        """
        file = filedialog.askopenfile()
        self.window.focus_force()   # for some reason the menu displays over the select window after opening the file dialog
        if file is not None:
            self._selected_file_l.config(text=file.name)
    
    def _select_model(self) -> None:
        """Calls the select function passing the selected file and closes the window.
        """
        self._select_btn.config(tk.DISABLED)
        self._select_fn(self._selected_file_l['text'])
        self.quit()
