import tkinter as tk

class Pixel:
    """
    A larger scale of a normal pixel, meaning that is a square of nxn real pixel, that can only take on gray scale.

    Attributes
    ----------

    Private:

        _x (float)
            The x coordinate of the center of the pixel (from left to right)
        _y (float)
            The y coordinate of the center of the pixel (from top to bottom)
        _size (int)
            The number of real pixels that compose the pixel object
        _canvas (Canvas)
            The canvas that contains the pixel
        _px (int)
            The ID of the rectangle displayed on the canvas
        _v (int)
            The gray value between 0 and 255 included
    
            
    Methods
    -------

    Public:

        __sub__
            The override of the __sub__ standard method, returns the distance per coordinate
        get_value
            Returns the gray value of the pixel
        switch_pixel
            Assigns a new gray to the pixel
    """

    def __init__(self, canvas: tk.Canvas, x: float, y: float, size: int) -> None:
        self._x: float = x
        self._y: float = y
        self._size: int = size
        self._canvas: tk.Canvas = canvas
        self._px: int = canvas.create_rectangle((x - size / 2, y - size / 2, x + size / 2, y + size / 2), fill='black', width=0)
        self._v: int = 0

    def __sub__(self, px) -> tuple[float, float]:
        return self._x - px._x, self._y - px._y

    def get_value(self) -> int:
        """Returns the gray value.

        Returns:
            int: The gray value between 0 and 255 included.
        """
        return self._v
    
    def switch_pixel(self, v: int) -> None:
        """Assigns the new given gray value.

        Args:
            v (int): The new gray value between 0 and 255 included.
        """
        self._v = v
        self._canvas.itemconfig(self._px, fill=f'#{int(v):02x}{int(v):02x}{int(v):02x}')
