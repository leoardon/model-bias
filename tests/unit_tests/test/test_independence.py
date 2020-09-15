import io
import unittest
import warnings

from mock import patch

from model_bias.test.independence import *

from ..utils import generate_not_independent_data


def _assert_not_print(test):
    @patch("sys.stdout", new_callable=io.StringIO)
    def _test(*args):
        args, mock = args[:-1], args[-1]
        test(*args)
        assert not mock.getvalue()

    return _test


class TestCase(unittest.TestCase):
    def setUp(self):
        self.data = generate_not_independent_data()

    @_assert_not_print
    def test_demographic_parity_with_values(self):

        y_hat = self.data[["y_hat"]].values
        z_protected = self.data[[f"protected_{i}" for i in range(1, 6)]].values

        p_value = demographic_parity(y_hat, z_protected)

        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    @_assert_not_print
    def test_demographic_parity_with_names(self):

        y_hat = "y_hat"
        z_protected = [f"protected_{i}" for i in range(1, 6)]

        p_value = demographic_parity(y_hat, z_protected, data=self.data)

        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    @_assert_not_print
    def test_demographic_parity_with_names_no_data(self):

        y_hat = "y_hat"
        z_protected = [f"protected_{i}" for i in range(1, 6)]

        with self.assertRaises(AssertionError):
            _p_value = demographic_parity(y_hat, z_protected)

    @_assert_not_print
    def test_equalized_odds_with_values(self):
        y_hat = self.data[["y_hat"]].values
        y_true = self.data[["y_true"]].values
        z_protected = self.data[[f"protected_{i}" for i in range(1, 6)]].values

        p_value = equalized_odds(y_hat, z_protected, y_true)

        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    @_assert_not_print
    def test_equalized_odds_with_names(self):

        y_hat = "y_hat"
        y_true = "y_true"
        z_protected = [f"protected_{i}" for i in range(1, 6)]

        p_value = equalized_odds(y_hat, z_protected, y_true, data=self.data)

        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    @_assert_not_print
    def test_equalized_odds_with_names_no_data(self):

        y_hat = "y_hat"
        y_true = "y_true"
        z_protected = [f"protected_{i}" for i in range(1, 6)]

        with self.assertRaises(AssertionError):
            _p_value = equalized_odds(y_hat, z_protected, y_true)
