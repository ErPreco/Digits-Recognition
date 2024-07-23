import tkinter as tk

class Window:
    """
    The base class for the windows of the "ui" module.

    Attributes
    ----------

    Public:

        window (Tk)
            The tkinter window
    
    
    Methods
    -------

    Public:
    
        get_window
            Returns the tkinter window
        quit
            Properly quits the window
        on_entry_click
            Cleans an entry of its placeholder
        on_focus_out
            Resets the placeholder of an entry and gives the focus to the window
    """

    def __init__(self, res: str=None, title: str=None, zoomed: bool=True) -> None:
        self.window: tk.Tk = tk.Tk()
        self.window.geometry(res)
        self.window.title(title)
        self.window.state('zoomed' if zoomed else None)

        self.window.protocol('WM_DELETE_WINDOW', self.quit)

    def get_window(self) -> tk.Tk:
        """Returns the tkinter window.

        Returns:
            Tk: The tkinter window.
        """
        return self.window
    
    def quit(self) -> None:
        """Properly quits the window.
        """
        self.window.quit()
        self.window.destroy()
    
    def on_entry_click(self, entry: tk.Entry, placeholder: str) -> None:
        """Cleans the given entry of its placeholder and sets the text color to black.

        Args:
            entry (Entry): The entry that triggered the event.
            placeholder (str): The entry placeholder.
        """
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.configure(foreground="black")

    def on_focus_out(self, entry: tk.Entry, placeholder: str) -> None:
        """Resets the placeholder of the given entry and gives the focus to the window.

        Args:
            entry (Entry): The entry that triggered the event.
            placeholder (str): The placeholder to reset.
        """
        if entry.get() == '':
            entry.insert(0, placeholder)
            entry.configure(foreground="gray")
        self.window.focus()
