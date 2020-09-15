from collections.abc import Iterable

import pandas as pd
from bokeh.io import output_file, output_notebook


def _in_notebook():  # pragma: no cover
    """Utility funciton to assess whether the code is running in a Jupyter notebook

    Returns:
    --------
    in_notebook (bool): True if the code is run in a Jupyter notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True

        return False
    except NameError:
        return False


def _retrieve_value(variable, data=None):
    """Helper method to retrive the values if only the name was passed
    """
    if isinstance(variable, (pd.Series, pd.DataFrame)):
        return variable

    if isinstance(variable, str) or (
        isinstance(variable, Iterable) and all(isinstance(i, str) for i in variable)
    ):
        assert data is not None, "`data` must be provided"
        return data[variable]

    if isinstance(variable, Iterable) and all(
        isinstance(i, pd.Series) for i in variable
    ):
        return pd.DataFrame({s.name: s for s in variable})

    raise ValueError(f"Unsupported type: {type(variable)}")


def _output(output_file_name):  # pragma: no cover
    """Output the plot
    """
    in_notebook = _in_notebook()
    if in_notebook:
        output_notebook(hide_banner=True)
    else:
        output_file(output_file_name)
