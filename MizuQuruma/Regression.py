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
    _display_info,
    display_warning,
    handle_single_feature,
)

from .model_selection import save_instance, load_instance
from .Metrics import RegressionMetrics


class LinearRegression:
    def __init__(self):
        self.X_train: np.ndarray = None
        self.y_train: np.ndarray = None
        self.num_iterations: int = 1
        self.learning_rate: float = 1e-3
        self.verbosity: int = 1
        self.weights_, self.bias_ = np.random.randn(2)

        # Initializaiton of internal Variables
        self.model_is_trained = False

        self.evaluation_metric_names = [
            "mse",
            "mae",
            "mean_squared_error",
            "mean_absolute_error",
        ]
        self.evaluation_metrics = RegressionMetrics
        self.evaluation_metric_mapping: dict = {
            name: getattr(self.evaluation_metrics, "calculate_" + name)
            for name in self.evaluation_metric_names
        }

    def _decorator_is_model_trained(
        _function: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        Decorator to check if the model is trained before executing a method.

        ---

        Description 📖:
        - This decorator checks whether the model has been trained using the `fit()` method before executing the decorated method.
        - If the model has not been trained, it raises a `RuntimeError` with an appropriate error message.

        ---

        Parameters ⚒️:
        - `_function (Callable[..., Any])`: The function to be decorated.

        ---

        Returns 📥:
            - `callable`: The wrapper function.

        ---

        Raises ⛔:
        - `RuntimeError`: If the model is not trained when calling the decorated function.

        ---

        Example 🎯:
        ```
        @_decorator_is_model_trained
        def predict(self, X: np.ndarray) -> np.ndarray:
            # Method implementation
        ```
        """

        def wrapper(self, *args, **kwargs):
            if not self.model_is_trained:
                error_message = f"First, train the model using the `fit()` method before using the `{_function.__name__}()` method."
                display_error(error_message=error_message, error_type=RuntimeError)

            return _function(self, *args, **kwargs)

        return wrapper

    def _display_info(self, message, min_verbosity=0):
        # Shows message based on minimal verbosity level set in the flag
        if self.verbosity >= min_verbosity:
            print(message)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        num_iterations: int = 100,
        learning_rate: int | float = 1e-3,
        verbosity: Literal[0, 1, 2] = 1,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """

        Fit the linear regression model to the training data.

        ---

        Description 📖:

        - This method fits the linear regression model to the provided training data using gradient descent optimization.
        It updates the model's weights and bias iteratively based on the calculated gradients until convergence or the specified number of iterations is reached.

        ---

        Parameters ⚒️:

        - `X_train` (np.ndarray): The feature matrix of shape (n_samples, n_features) containing the training data.
        - `y_train` (np.ndarray): The target values of shape (n_samples,) corresponding to the training data.
        - `num_iterations` (int): The number of iterations for which the gradient descent optimization will run. Defaults to 100.
        - `learning_rate` (int | float): The learning rate determining the step size in each iteration of gradient descent. Defaults to 1e-3.
        - `verbosity` (Literal[0,1,2]): Verbosity level controlling the amount of information displayed during training.
        - 0: No information displayed.
        - 1: Displays loss information after each iteration.
        - 2: Displays detailed information including iteration number and loss after each iteration. Defaults to 1.

        ---

        Returns 📤:

        - Tuple[float, np.ndarray, np.ndarray]: A tuple containing the final loss value, optimized weights, and bias of the linear regression model.

        ---

        Example 🎯:

        ```python
        import numpy as np
        from your_module import YourLinearRegressionModel

        # Assuming X_train and y_train are the training data
        model = YourLinearRegressionModel()
        loss, weights, bias = model.fit(X_train, y_train)
        ```

        """

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
                weight_gradients, bias_gradient = self._calculate_gradients(xi, yi)

                self.weights_ -= weight_gradients * self.learning_rate
                self.bias_ -= bias_gradient * self.learning_rate

            self.loss_ = self.evaluate(self.X_train, self.y_train)

            info_message = f"Iteration {iter_count}\tLoss: {self.loss_}"
            self._display_info(
                message=info_message,
                min_verbosity=2,
            )

        self.model_is_trained = True
        info_message = "Training is finished succesfully!"
        _display_info(info_message=info_message)

        self._display_info(
            message=f"Loss: {self.loss_}\nWeights: {self.weights_}\nBias: {self.bias_}",
            min_verbosity=1,
        )

        # self._display_info(f"Model is trained.\nFinal Loss:{self.loss_}", min_verbosity=1)
        return self.loss_, self.weights_, self.bias_

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        evaluation_metric: Literal["mse", "mae"] = "mse",
    ):
        """

        Evaluate the linear regression model on the test data using specified evaluation metric.

        ---

        Description 📖:

        - This method evaluates the performance of the linear regression model on the provided test data by comparing the predicted values with the actual target values.
        It calculates the evaluation metric specified by the user, such as Mean Squared Error (MSE) or Mean Absolute Error (MAE).

        ---

        Parameters ⚒️:

        - `X_test` (np.ndarray): The feature matrix of shape (n_samples, n_features) containing the test data.
        - `y_test` (np.ndarray): The target values of shape (n_samples,) corresponding to the test data.
        - `evaluation_metric` (Literal["mse", "mae"]): The evaluation metric used to assess the model's performance.
        - "mse": Mean Squared Error.
        - "mae": Mean Absolute Error. Defaults to "mse".

        ---

        Returns 📤:

        - float: The evaluation score indicating the performance of the model on the test data.

        ---

        Example 🎯:

        ```python
        import numpy as np
        from your_module import YourLinearRegressionModel

        # Assuming X_test and y_test are the test data
        model = YourLinearRegressionModel()
        evaluation_score = model.evaluate(X_test, y_test, evaluation_metric="mse")
        ```

        """

        y_pred = self.predict(X_test)
        y_real = y_test

        evaluation_calculator = self.evaluation_metrics(y_real=y_real, y_pred=y_pred)
        eval_function = self.evaluation_metric_mapping.get(evaluation_metric)

        return 1 - eval_function(evaluation_calculator)

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

        y_pred = self.predict(xi)  # (n_sample, n_feautures)
        weight_gradients = -2 * (yi - y_pred) * xi

        bias_gradient = -2 * (yi - y_pred)
        # don't multiply by anything because differentiation of a constant results in 1.

        return weight_gradients, bias_gradient

    def predict(self, X):
        """

        Predict target values using the trained linear regression model.

        ---

        Description 📖:

        - This method predicts the target values using the trained linear regression model based on the provided feature matrix.
            It utilizes the learned weights and bias of the model to make predictions.

        ---

        Parameters ⚒️:

        - `X` (np.ndarray): The feature matrix of shape (n_samples, n_features) for which predictions are to be made.

        ---

        Returns 📤:

        - np.ndarray: An array containing the predicted target values.

        ---

        Example 🎯:

        ```python
        import numpy as np
        from your_module import YourLinearRegressionModel

        # Assuming X_test is the feature matrix for prediction
        model = YourLinearRegressionModel()
        y_pred = model.predict(X_test)
        ```

        """

        return X * self.weights_ + self.bias_

    def generate_dataset(
        self,
        n_features: int = 1,
        n_samples: int = 10,
        coeff: Iterable[float | int] | None = None,
        intercept: float | None = None,
    ):
        """

        Generate a synthetic dataset for regression.

        ---

        Description 📖:

        - This method generates a synthetic dataset for regression tasks with specified features, samples, coefficients, and intercept.
        It randomly generates coefficients and intercept if not provided, and then generates feature and target arrays based on these parameters.

        ---

        Parameters ⚒️:
        
        - `n_features` (int): The number of features to generate. Defaults to 1.
        - `n_samples` (int): The number of samples to generate. Defaults to 10.
        - `coeff` (Iterable[float | int] | None): The coefficients to use for generating target values. If None, random coefficients are generated. Defaults to None.
        - `intercept` (float | None): The intercept to use for generating target values. If None, a random intercept is generated. Defaults to None.

        ---

        Returns 📤:

        - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the generated training feature matrix, training target values, testing feature matrix, and testing target values.

        ---

        Example 🎯:

        ```python
        import numpy as np
        from your_module import YourLinearRegressionModel

        # Assuming model is an instance of YourLinearRegressionModel
        X_train, y_train, X_test, y_test = model.generate_dataset(n_features=3, n_samples=100)
        ```

        """

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

        X_train, y_train = simultaneous_shuffle(X_train, y_train)
        X_test, y_test = simultaneous_shuffle(X_test, y_test)

        self._display_info(f"Generated Weights: {coeff}", min_verbosity=1)
        self._display_info(f"Generated Bias: {intercept}", min_verbosity=1)
        return X_train, y_train, X_test, y_test

    @_decorator_is_model_trained
    def save_model(self, filename="KNN_model.pkl") -> None:

        save_instance(self, filename=filename)

    def load_model(self, filename="KNN_model.pkl") -> Any:

        return load_instance(filename=filename)
