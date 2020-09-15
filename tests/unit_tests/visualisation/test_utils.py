import unittest

import pandas as pd

from model_bias.visualisation.utils import _retrieve_value

from ..utils import generate_not_independent_data


class TestCase(unittest.TestCase):
    def setUp(self):
        self.data = generate_not_independent_data()

    def test__retrieve_value_series(self):
        ret = _retrieve_value(self.data["y_hat"])
        self.assertIsInstance(ret, pd.Series)

    def test__retrieve_value_name(self):
        ret = _retrieve_value("y_hat", data=self.data)
        self.assertIsInstance(ret, pd.Series)

    def test__retrieve_value_name_no_data(self):
        with self.assertRaises(AssertionError):
            _ret = _retrieve_value("y_hat")

    def test__retrieve_value_dataframe(self):
        ret = _retrieve_value(self.data[["y_hat"]])
        self.assertIsInstance(ret, pd.DataFrame)

    def test__retrieve_value_names(self):
        ret = _retrieve_value(["y_hat", "y_true"], data=self.data)
        self.assertIsInstance(ret, pd.DataFrame)

    def test__retrieve_value_list_series(self):
        ret = _retrieve_value([self.data["y_hat"], self.data["y_true"]])
        self.assertIsInstance(ret, pd.DataFrame)

    def test__retrieve_value_raises_mixed_types(self):
        with self.assertRaises(ValueError):
            _ = _retrieve_value([self.data["y_hat"], "y_true"])
