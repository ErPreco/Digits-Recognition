import tkinter as tk

from ui.test import Pixel

class Canvas:
    """
    A canvas on which is possible drawing using the mouse.

    Attributes
    ----------

    Private:
    
        _canvas (Canvas)
            The tkinter canvas representing the object
        _px_size (int)
            The number of real pixels that compose the size of a pixel object
        _grid (list[list[Pixel]])
            The grid of pixels
    
    
    Methods
    -------

    Private:

        _init_grid
            Initializes the grid of pixels
        _on_draw
            Lits pixels to simulate drawing
    
    Public:

        get_canvas
            Returns the tkinter canvas
        res
            Returns the resolution of the canvas
        valid_pos
            Checks whether a point is inside the canvas
        clear
            Cleans the canvas as never touched
    """

    def __init__(self, window: tk.Tk, width: int, height: int, px_size: int) -> None:
        # creates the canvas
        self._canvas: tk.Canvas = tk.Canvas(window, width=width, height=height)
        self._canvas.pack()

        # creates the screen with dimmed pixels
        self._px_size = px_size
        self._grid: list[list[Pixel]] = []   # the grid of pixels as list of rows
        self._init_grid(width, height)

        # triggers _draw method when left button mouse is clicked and held
        self._canvas.bind('<B1-Motion>', self._on_draw)
    
    def _init_grid(self, width: int, height: int) -> None:
        """Initializes the grid of pixels by rows (_grid[i][j] means pixel of i-th row and j-th column).

        Args:
            width (int): The width of the canvas.
            height (int): The height of the canvas.
        """
        y = self._px_size / 2
        while y < height:
            row = []
            x = self._px_size / 2
            while x < width:
                row.append(Pixel(self._canvas, x, y, self._px_size))
                x += self._px_size
            self._grid.append(row)
            y += self._px_size

    def _on_draw(self, event: tk.Event) -> None:
        """Lits the pixels close to the mouse.

        Args:
            event (Event): The event of the mouse motion (left button).
        """
        brush_size = int(self._px_size * 5.5)   # harsh shapes can result if brush_size is a prefect multiple of _px_size
        xi, yi = event.x // self._px_size, event.y // self._px_size   # finds the indexes of the nearest pixel to the mouse
        if self.valid_pos(xi, yi):
            # whether the mouse is inside the screen:
            #   - considers the square of pixels of size brushSize around the mouse
            #   - checks which pixels are brushSize/2 away from the mouse starting from top-left and proceeding to the right
            #     (clamping if the coordinates exceed the canvas bounds)
            #   - if they are close enough, calculates the gray value
            px_offset = (brush_size / 2) // self._px_size
            for i in range(int(max(0, yi - px_offset)), int(min(self.res()[1], yi + px_offset + 1))):
                for j in range(int(max(0, xi - px_offset)), int(min(self.res()[0], xi + px_offset + 1))):
                    dst_x, dst_y = self._grid[i][j] - self._grid[yi][xi]
                    dst_x, dst_y = abs(dst_x), abs(dst_y)
                    if dst_x**2 + dst_y**2 <= (brush_size / 2)**2:
                        # if the pixel is close enough:
                        #   - sets the pixel gray value to 64
                        #   - blurs it with the 24 pixels around (5x5 exluding itself) using the weights matrix
                        near_pxs_sum = 0
                        px_weights = [[0.035, 0.035, 0.035, 0.035, 0.035],
                                      [0.035, 0.039, 0.039, 0.039, 0.035],
                                      [0.035, 0.039, 0.128, 0.039, 0.035],
                                      [0.035, 0.039, 0.039, 0.039, 0.035],
                                      [0.035, 0.035, 0.035, 0.035, 0.035]]
                        for ic in range(int(max(0, i - 2)), int(min(self.res()[1], i + 2 + 1))):
                            for jc in range(int(max(0, j - 2)), int(min(self.res()[0], j + 2 + 1))):
                                near_pxs_sum += 64 if ic == i and jc == j else self._grid[ic][jc].get_value() * px_weights[ic - (i - 2)][jc - (j - 2)]
                        self._grid[i][j].switch_pixel(near_pxs_sum if near_pxs_sum <= 255 else 255)

    def get_canvas(self) -> tk.Canvas:
        """Returns the tkinter canvas.

        Returns:
            Canvas: The tkinter canvas.
        """
        return self._canvas

    def get_grid(self) -> list[list[Pixel]]:
        """Returns the grid of pixels as list of rows.

        Returns:
            list[list[Pixel]]: The grid of pixels.
        """
        return self._grid

    def res(self) -> tuple[int, int]:
        """Returns the resolution of the canvas.

        Returns:
            tuple[int, int]: The tuple (width, height).
        """
        return len(self._grid[0]), len(self._grid)

    def valid_pos(self, x: int, y: int) -> bool:
        """Returns True whether the point (x, y) is inside the canvas, False otherwise.

        Args:
            x (int): The x coordinate of the point.
            y (int): The y coordinate of the point.

        Returns:
            bool: True whether (x, y) are inside the canvas, False otherwise.
        """
        valid_x = (x >= 0 and x < self.res()[0])
        valid_y = (y >= 0 and y < self.res()[1])
        return (valid_x and valid_y)

    def clean(self) -> None:
        """Dims all pixels of the canvas.
        """
        for i in range(self.res()[1]):
            for j in range(self.res()[0]):
                if self._grid[i][j].get_value() > 0:
                    self._grid[i][j].switch_pixel(0)
