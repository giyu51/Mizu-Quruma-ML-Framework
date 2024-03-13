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
    handle_single_feature,
)

from .LabelEncoders import StringToIntEncoder
from .model_selection import save_instance, load_instance
from .Metrics import ClassificationMetrics


class StochasticLinearRegression:
    def __init__(self):
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.num_iterations: int = 1
        self.learning_rate: float = 1e-3
        self.verbosity: int = 1
        self.weights_, self.bias_ = np.random.randn(2)

        # Initializaiton of internal Variables
        self.model_is_trained = False

    def display_info(self, message, min_verbosity=0):
        # Shows message based on minimal verbosity level set in the flag
        if self.verbosity >= min_verbosity:
            print(message)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_iterations: int = 100,
        learning_rate: int | float = 1e-3,
        verbosity: int = 1,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        #
        #
        #   """" ADJUST INFORMATION OUTPUT"""
        #

        # Validate types of passed arguments

        validate_types(
            variable=X_train,
            variable_name="X_train",
            desired_type=Iterable,
            function=self.fit,
        )

        validate_types(
            variable=y_train,
            variable_name="y_train",
            desired_type=Iterable,
            function=self.fit,
        )

        validate_types(
            variable=num_iterations,
            variable_name="num_iterations",
            desired_type=int,
            function=self.fit,
        )

        validate_types(
            variable=learning_rate,
            variable_name="learning_rate",
            desired_type=[int, float],
            function=self.fit,
        )

        validate_types(
            variable=verbosity,
            variable_name="verbosity",
            desired_type=int,
            function=self.fit,
        )

        # Ensure training data is in numpy format for integrity
        self.X_train = to_numpy(X_train)
        self.y_train = to_numpy(y_train)

        # If the shape of X_train is (n_samples, )
        # it will turn it to (n_samples, 1) which will correspond to
        # format (n_samples, n_features)

        self.X_train = handle_single_feature(self.X_train)

        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.verbosity = verbosity

        for iter_count in range(self.num_iterations):
            for xi, yi in zip(self.X_train, self.y_train):
                # ic(xi)
                # ic(yi)
                # ic("_")
                weight_gradients, bias_gradient = self._calculate_gradients(xi, yi)

                self.weights_ -= weight_gradients * self.learning_rate
                self.bias_ -= bias_gradient * self.learning_rate

            self.loss_ = self.evaluate(self.X_train, self.y_train)

            info_message = f"Iteration {iter_count}\tLoss: {self.loss_}"
            self.display_info(
                message=info_message,
                min_verbosity=2,
            )

        self.model_is_trained = True
        info_message = "Training is finished succesfully!"
        display_info(info_message=info_message)

        self.display_info(
            message=f"Loss{self.loss_}\nWeights: {self.weights_}\nBias: {self.bias_}",
            min_verbosity=1,
        )

        # self.display_info(f"Model is trained.\nFinal Loss:{self.loss_}", min_verbosity=1)
        return self.loss_, self.weights_, self.bias_

    def _mse(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.square(y - y_pred))

    def _mae(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    def evaluate(self, X, y, evaluation_metric: Literal["mse", "mae"] = "mse"):
        self.evaluation_metric = evaluation_metric_names.get(evaluation_metric)
        return self._mse(X, y)

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
        return X * self.weights_ + self.bias_

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
