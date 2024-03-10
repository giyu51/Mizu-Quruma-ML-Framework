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
        self.learning_rate = 1e-3
        self.X_train = None
        self.y_train = None
        self.num_iterations = None
        self.verbosity = 1
        self.weights_, self.bias_ = np.random.randn(2)

    def display_info(self, message, min_verbosity=0):
        if self.verbosity >= min_verbosity:
            print(message)

    def fit(
        self, X_train, y_train, num_iterations=100, learning_rate=1e-3, verbosity=1
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbosity = verbosity

        for iter_count in range(num_iterations):
            self.loss_ = self.evaluate(self.X_train, self.y_train)
            self.display_info(
                message=f"Iteration {iter_count}\tLoss: {self.loss_}",
                min_verbosity=2,
            )
            for xi, yi in zip(self.X_train, self.y_train):
                # ic(xi)
                # ic(yi)
                # ic("_")
                weight_gradients, bias_gradient = self._calculate_gradients(xi, yi)

                self.weights_ -= weight_gradients * self.learning_rate
                self.bias_ -= bias_gradient * self.learning_rate

        self.display_info(message=f"\nWeights: {self.weights_}\nBias: {self.bias_}")
        # self.display_info(f"Model is trained.\nFinal Loss:{self.loss_}", min_verbosity=1)
        return self.loss_, self.weights_, self.bias_

    def _mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.square(y - y_pred))

    def _mae(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    def evaluate(self, X, y):
        return self._mse(X, y)

    def validate_feature_format(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        return X

    def _calculate_gradients(self, xi: np.ndarray, yi: np.ndarray):
        y_pred = self.predict(xi)

        # dy/dw1 = -2 * (y - (X1 * w1 + X2 * w2 + b)) * (W1)
        # dy/dw2 = -2 * (y - (X1 * w1 + X2 * w2 + b)) * (W2)
        # dy/db = -2 * (y - (X1 * w1 + X2 * w2 + b))

        # [x1, x2]
        #    x
        # [w1, w2]
        #    +
        #   [b]

        wi = self.weights_

        y_pred = self.predict(xi)  # (n, 2)
        weight_gradients = -2 * (yi - y_pred) * xi

        bias_gradient = -2 * (yi - y_pred)
        # don't multiply by anything because differentiation of a constant results in 1.

        return weight_gradients, bias_gradient

    def predict(self, X):
        return self._get_linear_combination(X=X, weights=self.weights_, bias=self.bias_)

    def _get_linear_combination(self, X, weights, bias):
        return X * weights + bias

    def generate_dataset(
        self,
        n_features: int = 1,
        n_samples: int = 10,
        coeff: Iterable[float | int] | None = None,
        intercept: float | None = None,
    ):

        if coeff is None:
            coeff = np.random.rand(n_features) * np.random.randint(1, 10)
        else:
            coeff = to_numpy(coeff)
            if coeff.size != n_features:
                message = f"The shape of coefficients (coeff) should be (n_features,). `n_features` is {n_features} and shape of `coeff` is {coeff.shape}"
                display_error(error_message=message, error_type=ValueError)
        if intercept is None:
            intercept = np.random.randn(1) * np.random.randint(1, 10)

        X_train = np.random.random(size=(n_samples, n_features))
        X_test = np.random.random(size=(n_samples, n_features))

        y_train = X_train * coeff + intercept
        y_test = X_test * coeff + intercept

        ic(f"Weights: {coeff}")
        ic(f"Bias: {intercept}")
        return X_train, y_train, X_test, y_test
