import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from models.metrics import print_metrics_regression, mean_squared_logarithmic_error


def test_print_metrics_regression_returns_six_metrics():
    y_true = np.array([1, 2, 3])
    preds = np.array([1, 2, 3])
    results = print_metrics_regression(y_true, preds, verbose=0)
    assert isinstance(results, list)
    assert len(results) == 6


def test_mean_squared_logarithmic_error_positive_inputs():
    y_true = np.array([1, 2, 3])
    preds = np.array([1, 2, 3])
    # Should not raise and should return zero for identical arrays
    msle = mean_squared_logarithmic_error(y_true, preds)
    assert msle == 0

