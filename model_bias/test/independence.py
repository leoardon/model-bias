import os
import sys
from collections.abc import Iterable
from contextlib import contextmanager

from CCIT.CCIT import CCIT as _test

#############################################
# Helpers
#############################################


@contextmanager
def _disable_print():
    """ Disable print statement from external packages
    """
    prev = sys.stdout
    sys.stdout = open(os.devnull, "w")
    yield
    sys.stdout = prev


def _retrieve_values(variable, data=None):
    """Helper method to retrive the values if only the name was passed
    """
    if isinstance(variable, str) or (
        isinstance(variable, Iterable) and all(isinstance(i, str) for i in variable)
    ):
        assert data is not None, "`data` must be provided"
        variable = [variable] if isinstance(variable, str) else variable
        return data[variable].values

    return variable


def _test_wrapper(x_variable, y_variable, z_variable=None):
    """ Method to wrap the actual test to run bootstrat if the dataset
    is too small
    """
    bootstrap = len(x_variable) < 1000
    with _disable_print():
        return _test(
            x_variable, y_variable, z_variable, bootstrap=bootstrap, num_iter=30
        )


#############################################


def demographic_parity(y_pred, z_protected, data=None):
    """ Perform a statistical test evaluating whether the Demographic Parity
    measure is satisfied.

    Params:
    -------
    y_pred (str or [str] or np.ndarray): the predicted variable
    z_protected (str or [str] or np.ndarray): the protected variable
    data (pd.DataFrame): the data to retrive the variables from

    Returns:
    --------
    p_value (float): the p_value of the hypothesis test with
                        H_0: y_pred independent of z_protected

    Raises:
    -------
    AssertionError: if the variable names are passed without the data
    """
    y_pred = _retrieve_values(y_pred, data=data)
    z_protected = _retrieve_values(z_protected, data=data)

    return _test_wrapper(y_pred, z_protected)


def equalized_odds(y_pred, z_protected, y_true, data=None):
    """Perform a statistical test evaluating whether the Demographic Parity
    measure is satisfied.

    Params:
    -------
    y_pred (str or [str] or np.ndarray): the predicted variable
    z_protected (str or [str] or np.ndarray): the protected variable
    y_true (str or [str] or np.ndarray): the true variable
    data (pd.DataFrame): the data to retrive the variables from

    Returns:
    --------
    p_value (float): the p_value of the hypothesis test with
                        H_0: y_pred independent of z_protected given y_true

    Raises:
    -------
    AssertionError: if the variable names are passed without the data
    """
    y_pred = _retrieve_values(y_pred, data=data)
    z_protected = _retrieve_values(z_protected, data=data)
    y_true = _retrieve_values(y_true, data=data)

    return _test_wrapper(y_pred, z_protected, y_true)
