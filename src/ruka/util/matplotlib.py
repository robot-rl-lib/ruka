import matplotlib.pyplot as plt

from ruka_os import in_cloud

from . import distributed_fs as dfs


def plot(path):
    """
    - plt.plot() if in jupyter notebook;
    - plt.savefig(path) + dfs.upload_maybe(path) otherwise.
    """
    if is_jupyter():
        plt.plot()
    else:
        plt.savefig(path)
        dfs.upload_maybe(path)


def is_jupyter() -> bool:
    """
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter