"""Set the PYTHONPATH to be able to import the code to test
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model_bias"))
)

import model_bias
