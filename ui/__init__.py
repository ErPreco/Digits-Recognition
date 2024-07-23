import tkinter as tk

from .window import Window

def exec(window: tk.Tk) -> None:
    """Executes the tkinter window given.
    """
    window.mainloop()
