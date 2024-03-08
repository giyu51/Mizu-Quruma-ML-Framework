import numpy as np
from icecream import ic
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Literal, List, Dict, Callable, Any, Iterable, Tuple
from ctypes import ArgumentError
from .internal_methods import (
    validate_types,
    validate_shapes,
    to_numpy,
    simultaneous_shuffle,
    display_error,
    display_info,
    display_warning,
)

from .LabelEncoders import StringToIntEncoder
from .model_selection import save_instance, load_instance
from .Metrics import ClassificationMetrics


class StochasticLinearRegression:
    def __init__(self):
        self.learning_rate = None
        self.X_train = None
        self.y_train = None
        self.num_iterations = None
        self.coeff_, self.intercept_ = np.random.randn(2)

    def fit(self, X_train, y_train, num_iterations=3, learning_rate=1e-3):
        self.X_train = X_train
        self.y_train = y_train
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate

        for _ in range(num_iterations):
            for xi, yi in zip(self.X_train, self.y_train):
                gradients = self._calculate_gradients(xi, yi)
                self.coeff_ -= gradients[0] * self.learning_rate
                self.intercept_ -= gradients[1] * self.learning_rate

        return self.coeff_, self.intercept_

    def _mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.square(y - y_pred))

    def _mae(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    def evaluate(self, X, y):
        return self._mse(X, y)

    def _calculate_gradients(self, xi: float, yi: float):
        y_pred = self.predict(xi)
        dk = -2 * np.mean(yi - y_pred) * xi
        db = -2 * np.mean(
            yi - y_pred
        )  # don't multiply by anything because differentiation of a constant results in 1.

        return dk, db

    def predict(self, X):
        return X * self.coeff_ + self.intercept_
