def print_progress_bar (iteration: int, total: int, prefix: str='', suffix: str='', decimals: int=1, length: int=100, fill: str='█', print_end: str='\n') -> None:
    """Prints a progress bar on terminal.

    Args:
        iteration (int): The reached progress out of total.
        total (int): The total of the porgress.
        prefix (str): A short description of the progress bar (wrote before the bar).
        suffix (str): A short description of the complete progress (wrote after the bar).
        decimals (int): The number of decimals of the percentage.
        length (int): The length of the bar.
        fill (str): The character that represents the filled bar.
        print_end (str): The end character printed at the complete progress.
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix}{' ' if prefix != '' else ''}|{bar}| {percent}% {suffix}', end='\r' if iteration != total else '')
    
    if iteration == total: 
        print(end=print_end)
