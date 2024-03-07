import numpy as np
from collections.abc import Iterable
from typing import Literal, Tuple
import warnings
import matplotlib.pyplot as plt
import time
import pickle


class LinearRegression:
    def __init__(self) -> None:
        self.verbosity = 1
        self.modelIsTrained = False
        self.loss_function = self.mse

    def fit(
        self,
        X: Iterable[int | float],
        Y: Iterable[int | float],
        loss_function: Literal["mse", "mae"] = "mse",
        max_iterations: int = 5000,
        learning_rate: float = 1e-3,
        loss_threshold: float = 1e-7,
        d_slope: float = 1e-5,
        d_intercept: float = 1e-5,
        initial_slope: None | float | int = None,
        initial_intercept: None | float | int = None,
        verbosity: Literal[0, 1] = 1,
    ) -> float:
        """
        Fit a linear regression model to the provided data.

        Parameters:
        - X (Iterable[int | float]): The input data as a list of integers or floats.
        - Y (Iterable[int | float]): The target output data as a list of integers or floats.
        - loss_function (Literal["mse", "mae"], optional): The loss function to use, either "mse" (Mean Squared Error) or "mae" (Mean Absolute Error). Defaults to "mse".
        - max_iterations (int, optional): The maximum number of iterations for training. Defaults to 5000.
        - learning_rate (float, optional): The learning rate for gradient descent. Defaults to 1e-3.
        - loss_threshold (float, optional): The threshold for stopping training when the loss is below this value. Defaults to 1e-7.
        - d_slope (float, optional): The increment for calculating the slope derivative. Defaults to 1e-5.
        - d_intercept (float, optional): The increment for calculating the intercept derivative. Defaults to 1e-5.
        - initial_slope (None | float | int, optional): Initial value for the slope. If None, a random value is used. Defaults to None.
        - initial_intercept (None | float | int, optional): Initial value for the intercept. If None, a random value is used. Defaults to None.
        - verbosity (Literal[0, 1], optional): Verbosity level for output messages. Defaults to 1.

        Returns:
        - float: The final loss value after training.

        Note: Use `predict()` and `evaluate()` for predictions and evaluations.
        """
        # Convert input data to NumPy arrays
        self.X = np.array(X, dtype=np.float32) if not isinstance(X, np.ndarray) else X
        self.Y = np.array(Y, dtype=np.float32) if not isinstance(Y, np.ndarray) else Y

        # Validate the data
        self.validate_data(self.X, self.Y)

        # Determine the loss function based on the user's choice
        if loss_function.lower() == "mse":
            self.loss_function = self.mse
        elif loss_function.lower() == "mae":
            self.loss_function = self.mae
        else:
            raise ValueError(f"Loss function is not availabe: {loss_function}")

        self.loss_history = []

        # Initialize model parameters
        self.predicted_slope = (
            np.random.random() if initial_slope is None else initial_slope
        )
        self.predicted_intercept = (
            np.random.random() if initial_intercept is None else initial_intercept
        )
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.d_slope = d_slope
        self.d_intercept = d_intercept
        self.loss_threshold = loss_threshold
        self.verbosity = verbosity
        self.iteration_count = 0
        self.modelIsTrained = False

        # Training loop
        while self.iteration_count < self.max_iterations and not (self.modelIsTrained):
            for _ in range(len(self.X)):
                slope_gradient, intercept_gradient = self.gradient_descend()

                # Check if the loss is below the threshold
                if self.loss < self.loss_threshold:
                    self.modelIsTrained = True
                    break

                # Update model parameters using gradient descent
                self.predicted_slope = (
                    self.predicted_slope - slope_gradient * self.learning_rate
                )
                self.predicted_intercept = (
                    self.predicted_intercept - intercept_gradient * self.learning_rate
                )

            self.loss_history.append(self.loss)

            self.iteration_count += 1

        self.modelIsTrained = True
        self.log_message(f"Training finished in {self.iteration_count} iterations")

        if self.iteration_count == self.max_iterations:
            self.log_message(
                f"Reached the maximum iteration limit of {self.max_iterations}. If the loss is still too high, consider increasing the `max_iterations` argument."
            )

        return self.loss

    def log_message(self, message: str, warning: bool = False):
        """
        Log a message or warning.

        Parameters:
        - message (str): The message to be logged.
        - warning (bool, optional): Whether the message is a warning. Defaults to False.

        Note: This method is intended for internal use and should not be called separately.
        """
        if self.verbosity > 0:
            if not warning:
                # Output message and add the line below for better readability
                mesage = str(mesage) if not isinstance(message, str) else message
                print(message, f"\n{'-' * len(message)}")
            else:
                warnings.warn(message=message)

    def validate_data(self, X: Iterable[float | int], Y: Iterable[float | int] = None):
        """
        Validate the shapes and properties of the input data.

        Parameters:
        - X (Iterable[float | int]): The input data as a list of integers or floats.
        - Y (Iterable[float | int], optional): The target output data as a list of integers or floats.

        Raises:
        - ValueError: If the shapes of X and Y do not match or other data validation conditions are not met.
        """
        # Get the shapes of X and Y
        X_shape = X.shape
        if Y is not None:
            Y_shape = Y.shape

            # Check if the shapes of X and Y match
            if X_shape != Y_shape:
                raise ValueError(
                    f"The shapes of X and Y must match.\nShape of X: {X_shape}\nShape of Y: {Y_shape}"
                )
            if len(Y_shape) == 0:
                raise ValueError("The shape of Y cannot be zero.")
            if not isinstance(Y, Iterable):
                raise ValueError("Y is not an iterable object.")

            # Check if the dataset is very small and may lead to inaccuracies
            if Y_shape[0] < 3:
                self.log_message(
                    "WARNING: The dataset is very small, which can lead to inaccuracies.",
                    warning=True,
                )
        if len(X_shape) == 0:
            raise ValueError("The shape of X cannot be zero.")

        if not isinstance(X, Iterable):
            raise ValueError("X is not an iterable object.")

    def mse(
        self,
        slope: float,
        intercept: float,
        X: None | Iterable[float | int] = None,
        Y: None | Iterable[float | int] = None,
    ) -> float:
        """
        Calculate the Mean Squared Error (MSE) loss for the given parameters.

        Parameters:
        - slope (float): The slope value of the linear regression model.
        - intercept (float): The intercept value of the linear regression model.
        - X (None | Iterable[float | int], optional): The input data. Defaults to None (uses internal X data).
        - Y (None | Iterable[float | int], optional): The target output data. Defaults to None (uses internal Y data).

        Returns:
        - float: The MSE loss value.

        Note: This method is intended for internal use and should not be called separately.
        """
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y
        # Calculate the predicted values using the provided slope and intercept
        prediction = X * slope + intercept

        # Calculate the mean squared error loss
        loss = np.mean(np.square(Y - prediction))
        return loss

    def mae(
        self,
        slope: float,
        intercept: float,
        X: None | Iterable[float | int] = None,
        Y: None | Iterable[float | int] = None,
    ) -> float:
        """
        Calculate the Mean Absolute Error (MAE) loss for the given parameters.

        Parameters:
        - slope (float): The slope value of the linear regression model.
        - intercept (float): The intercept value of the linear regression model.
        - X (None | Iterable[float | int], optional): The input data. Defaults to None (uses internal X data).
        - Y (None | Iterable[float | int], optional): The target output data. Defaults to None (uses internal Y data).

        Returns:
        - float: The MAE loss value.

        Note: This method is intended for internal use and should not be called separately.
        """
        # Calculate the predicted values using the provided slope and intercept
        prediction = X * slope + intercept

        # Calculate the mean absolute error loss
        loss = np.mean(np.abs(Y - prediction))
        return loss

    def gradient_descend(self):
        """
        Perform gradient descent and calculate gradients for slope and intercept.

        Returns:
        - slope_gradient (float): The gradient of the slope.
        - intercept_gradient (float): The gradient of the intercept.

        Note: This method is intended for internal use and should not be called separately.
        """
        # update the loss value
        self.loss = self.loss_function(self.predicted_slope, self.predicted_intercept)
        slope_func_increment = (
            self.loss_function(
                self.predicted_slope + self.d_slope, self.predicted_intercept
            )
            - self.loss
        )
        intercept_func_increment = (
            self.loss_function(
                self.predicted_slope, self.predicted_intercept + self.d_intercept
            )
            - self.loss
        )

        slope_arg_increment = self.d_slope
        intercept_arg_increment = self.d_intercept

        slope_gradient = slope_func_increment / slope_arg_increment
        intercept_gradient = intercept_func_increment / intercept_arg_increment

        return slope_gradient, intercept_gradient

    def predict(
        self, X: Iterable[float | int], verbosity: Literal[0, 1] = 1
    ) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters:
        - X (Iterable[float | int]): The input data for making predictions.
        - verbosity (Literal[0, 1], optional): Verbosity level for output messages. Defaults to 1.

        Returns:
        - np.ndarray: The predicted values.

        Note: Use for making predictions.


        """
        self.verbosity = verbosity
        if self.modelIsTrained:
            start_time = time.perf_counter()
            X = np.array(X, dtype=np.float32) if not isinstance(X, np.ndarray) else X
            self.validate_data(X=X)

            prediction = self.predicted_slope * X + self.predicted_intercept

            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            self.log_message(f"Elapsed time: {elapsed_time:.5f}")

            return prediction
        raise ValueError(
            "The model has not been trained yet. Use `fit()` method to train the model"
        )

    def evaluate(
        self,
        X: Iterable[float | int],
        Y: Iterable[float | int],
        verbosity: Literal[0, 1] = 1,
    ) -> float:
        """
        Evaluate the model's performance by calculating the loss on a given dataset.

        Parameters:
        - X (Iterable[float | int]): The input data as an iterable of floats or integers.
        - Y (Iterable[float | int]): The target output data as an iterable of floats or integers.
        - verbosity (Literal[0, 1], optional): Verbosity level for logging messages. Defaults to 1.

        Returns:
        - float: The calculated loss value.

        Note: This method is intended for users to assess the model's performance.
        """
        self.verbosity = verbosity
        if self.modelIsTrained:
            X = np.array(X, dtype=np.float32) if not isinstance(X, np.ndarray) else X
            Y = np.array(Y, dtype=np.float32) if not isinstance(Y, np.ndarray) else Y
            self.validate_data(X, Y)

            loss = self.loss_function(
                slope=self.predicted_slope, intercept=self.predicted_intercept, X=X, Y=Y
            )
            return loss
        raise ValueError(
            "The model has not been trained yet. Use `fit()` method to train the model"
        )

    def plot_loss(
        self, figure_size: Tuple[int, int] = (6, 4), save_name: None | str = None
    ) -> None:
        """
        Plot the training loss history over iterations.

        Parameters:
        - figure_size (Tuple[int, int], optional): The size of the figure for the plot. Defaults to (6, 4).
        - save_name (str, optional): The name of the file to save the plot. If not provided, the plot is displayed but not saved.

        Note: This method is used to visualize the training progress by plotting the loss history over iterations.
        """
        if self.modelIsTrained:
            plt.figure(figsize=figure_size)
            plt.plot(range(self.iteration_count), self.loss_history)
            plt.grid(True)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")

            if isinstance(save_name, str):
                plt.savefig(save_name)

            plt.show()
        else:
            raise ValueError(
                "The model has not been trained yet. Use `fit()` method to train the model"
            )

    def save_model(self, save_path: str):
        """
        Save the trained model to a file (.vf).

        Parameters:
        - save_path (str): The path where the model will be saved (path/model_name).

        Note: This method is for saving the trained model's parameters for later use.
        """
        if self.modelIsTrained:
            model_data = {
                "slope": self.predicted_slope,
                "intercept": self.predicted_intercept,
            }
            try:
                with open(save_path, "wb") as file:
                    pickle.dump(model_data, file)
            except Exception as err:
                raise SystemError(f"Error while saving the model: {err}")

            self.log_message(f"Model {save_path} successfully saved!")
        else:
            raise ValueError(
                "The model has not been trained yet. Use `fit()` method to train the model"
            )

    def load_model(self, model_name: str):
        """
        Load a previously saved model (.vf) from a file.

        Parameters:
        - model_name (str): The name of the saved model file.

        Note: This method is for loading a trained model's parameters for reuse.
        """
        try:
            with open(model_name, "rb") as file:
                model_data = pickle.load(file)
                self.predicted_slope = model_data["slope"]
                self.predicted_intercept = model_data["intercept"]
                self.modelIsTrained = True
                self.log_message(f"Model {model_name} successfully loaded!")

        except Exception as err:
            raise SystemError(f"Error while loading the model: {err}")
