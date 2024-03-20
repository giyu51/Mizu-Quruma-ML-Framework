import numpy as np

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


class KNN:

    def __init__(
        self,
        K: int = 2,
        neighbour_number: int | None = None,
        verbose: Literal[0, 1, 2] = 1,
        regression_mode: bool = False,
        # 0: No output except for critical errors.
        # 1: Standard output.
        # 2: Output including validation checks.
    ) -> None:
        """
        K-Nearest Neighbors (KNN) classifier implementation.

        ---

        Description 📖:
        - This class implements the K-Nearest Neighbors algorithm for classification or regression.
        - If `regression_mode` is set to True, the algorithm performs regression instead of classification.
        - For regression mode, it predicts the target value for a data point based on the average (or weighted average) of the target values of its K nearest neighbors.
        - Various evaluation metrics can be calculated to assess the performance of the classifier/regressor.

        ---

        Parameters ⚒️:
        - `K (int)`: Number of neighbors to consider when making predictions.
        - `neighbour_number (int | None)`: Optional parameter specifying the number of neighbors. If provided, it must have the same value as `K`.
        - `verbose (Literal[0, 1, 2])`: Verbosity level of the algorithm's output.
        - 0: No output except for critical errors.
        - 1: Standard output.
        - 2: Output including validation checks.
        - `regression_mode (bool)`: Flag to indicate whether the algorithm should operate in regression mode. Default is False, indicating classification mode.

        ---

        Returns 📥:
        - None

        ---

        Raises ⛔:
        - `ValueError`: If `K` and `neighbour_number` attributes have different values.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1, regression_mode=False)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.fit(X_train, y_train)

        # Make predictions
        X_test = np.array([[4, 5], [5, 6]])
        predictions = knn.predict(X_test)

        print(predictions)
        ```
        """
        # Initialization of variables
        self.K: int = K
        self.verbose: Literal[0, 1, 2] = verbose
        self.neigbour_number: int | None = neighbour_number
        self.regression_mode: bool = regression_mode
        # Validation of Variable Types

        self._validate_types(
            variable=K, variable_name="K", desired_type=int, function=self.__init__
        )

        self._validate_types(
            variable=neighbour_number,
            variable_name="neighbour_number",
            desired_type=[int, None],
            function=self.__init__,
        )

        self._validate_types(
            variable=verbose,
            variable_name="verbose",
            desired_type=Literal[0, 1, 2],
            function=self.__init__,
        )

        self._validate_types(
            variable=regression_mode,
            variable_name="regression_mode",
            desired_type=bool,
            function=self.__init__,
        )

        # Handle the case if both `K` and `neighbour_number` provided but the values of them are different.
        if self.neigbour_number is not None and self.K != self.neigbour_number:
            error_message = (
                "`K` and `neighbour_number` attributes must have the same value."
            )
            display_error(error_message=error_message, error_type=ValueError)

        self.evaluation_metric_names: List[str] = [
            "accuracy",
            "confusion_matrix",
            "confusion_metrics",
            "sensitivity",
            "specificity",
            "recall",
            "precision",
            "f1_score",
        ]

        self.clfMetrics = ClassificationMetrics
        self.evaluation_metric_mapping: dict = {
            name: getattr(self.clfMetrics, "calculate_" + name)
            for name in self.evaluation_metric_names
        }

        # Indicate that the model has not been trained yet
        self.model_is_trained: bool = False

        # Initialization of training data containers
        self.X_train = self.y_train = np.array([])
        self.inputs, self.outputs = self.X_train.view(), self.y_train.view()

        # Initialization of Encoder
        self.label_encoder = StringToIntEncoder()

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

    def _display_guide_message(self, message):
        """
        Internal method to display a guide message based on the verbosity level.

        ---

        Description 📖:
        - This internal method displays a guide message based on the verbosity level (`verbose` attribute) of the class.
        - It is used to provide informational messages to the user during the execution of the algorithm.

        ---

        Parameters ⚒️:
        - `message (str)`: The message to be displayed.

        ---

        Returns 📥:
        - None

        ---

        Example 🎯:
        ```
        _display_guide_message("This is a guide message.")
        ```
        """
        if self.verbose == 2:
            print(message)

    def _display_info_message(self, info_message: str):
        """
        Internal method to display an informational message based on the verbosity level.

        ---

        Description 📖:
        - This internal method displays an informational message based on the verbosity level (`verbose` attribute) of the class.
        - It is used to provide important information to the user during the execution of the algorithm.

        ---

        Parameters ⚒️:
        - `info_message (str)`: The informational message to be displayed.

        ---

        Returns 📥:
        - None

        ---

        Example 🎯:
        ```
        message = "This is an informational message."
        _display_info_message(message)
        ```
        """
        if self.verbose > 0:
            display_info(info_message=info_message)

    def show_workflow(self):
        """
        Display the step-by-step workflow of using the K-Nearest Neighbours Algorithm.

        ---

        Description 📖:
        - This method provides a brief overview of the typical workflow involved in using the KNN classifier.
        - It outlines the main steps required from initiating the class to saving and loading the trained model.

        ---

        Parameters ⚒️:
        - None

        ---

        Returns 📥:
        - None

        ---

        Raises ⛔:
        - None

        ---

        Example 🎯:
        ```
        # Display the workflow
        knn.show_workflow()
        ```

        """

        workflow = "[ K-Nearest Neighbours Algorithm workflow: ]\n\n1. Initiate the class: Start by creating an instance of the KNN classifier.\n\n2. fit(): Train the model by providing training data using the `fit()` method.\n\n3. evaluate(): Assess the model's performance using evaluation metrics and testing data with the `evaluate()` method.\n\n4. predict(): Optionally, make predictions for new data points using the `predict()` method.\n\n5. plot_data(): Visualize the data, including training and testing points, and decision boundaries if desired, using the `plot_data()` method.\n\n6. save_model() and load_model(): Save the trained model for future use with `save_model()` and load a saved model using `load_model()`."

        self._display_info_message(workflow)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the KNN classifier to the training data.

        ---

        Description 📖:
        - This method fits (trains) the KNN classifier to the provided training data.
        - It validates the shapes and types of the input and output arrays before training the model.
        - The encoder is updated to remember the whole vocabulary of class labels.
        - After training, a success message is displayed if the verbosity level is greater than 0.

        ---

        Parameters ⚒️:
        - `X_train (np.ndarray)`: The input feature array for training.
        - `y_train (np.ndarray)`: The target output array for training.

        ---

        Returns 📥:
        - None

        ---

        Raises ⛔:
        - `ValueError`: If the shapes of `X_train` and `y_train` are incompatible.
        - `TypeError`: If the types of `X_train` or `y_train` are not iterable.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.fit(X_train, y_train)
        ```
        """

        self._validate_shapes(X_train, y_train, metrics="check_sample_count")
        self._validate_types(
            variable=X_train,
            variable_name="X_train/inputs",
            desired_type=Iterable,
            function=self.fit,
        )
        self._validate_types(
            variable=y_train,
            variable_name="y_train/outputs",
            desired_type=Iterable,
            function=self.fit,
        )

        # Update class variables accordingly to input and output arrays
        self.X_train = to_numpy(X_train)
        self.y_train = to_numpy(y_train)

        # Let encoder remember the whole vocabulary
        self.label_encoder.fit(vocabulary=self.y_train)

        self.model_is_trained = True

        info_message = "✅ Model is trained (fitted) succesfully ✅"
        self._display_info_message(info_message)

    def train_the_model(self, inputs: np.ndarray, outputs: np.ndarray):
        """
        Train the KNN classifier using the provided input and output arrays.

        ---

        Description 📖:
        - This method trains the KNN classifier using the provided input and output arrays.
        - It is equivalent to calling the `fit()` method with the same parameters.

        ---

        Parameters ⚒️:
        - `inputs (np.ndarray)`: The input feature array for training.
        - `outputs (np.ndarray)`: The target output array for training.

        ---

        Returns 📥:
        - None

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.train_the_model(inputs=X_train, outputs=y_train)
        ```
        """
        return self.fit(X_train=inputs, y_train=outputs)

    def _encode(self, X: np.ndarray) -> np.ndarray:
        """
        Internally used method to encode class labels.

        ---

        Description 📖:
        - This method internally utilizes the label encoder to encode class labels in the provided array `X`.
        - It returns the encoded array.

        ---

        Parameters ⚒️:
        - `X (np.ndarray)`: Array containing class labels to be encoded.

        ---

        Returns 📥:
        - `np.ndarray`: Encoded array containing the encoded class labels.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array(['A', 'B', 'A'])

        # Encode class labels
        encoded_labels = knn._encode(X_train)
        ```
        """
        return self.label_encoder.encode(X)

    def _decode(self, X: np.ndarray) -> np.ndarray:
        """
        Internally used method to decode encoded class labels.

        ---

        Description 📖:
        - This method internally utilizes the label encoder to decode encoded class labels in the provided array `X`.
        - It returns the decoded array.

        ---

        Parameters ⚒️:
        - `X (np.ndarray)`: Array containing encoded class labels to be decoded.

        ---

        Returns 📥:
        - `np.ndarray`: Decoded array containing the original class labels.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array(['A', 'B', 'A'])

        # Encode class labels
        encoded_labels = knn._encode(y_train)

        # Decode class labels
        decoded_labels = knn._decode(encoded_labels)
        ```
        """
        return self.label_encoder.decode(X)

    @_decorator_is_model_trained
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metrics: List[
            Literal[
                "accuracy",
                "confusion_matrix",
                "confusion_metrics",
                "sensitivity",
                "specificity",
                "recall",
                "precision",
                "f1_score",
            ]
        ] = ["accuracy"],
    ) -> Dict:
        """
        Evaluate the performance of the trained KNN classifier on the test data.

        ---

        Description 📖:
        - This method evaluates the performance of the trained KNN classifier on the provided test data.
        - It computes various evaluation metrics, such as accuracy, confusion matrix, sensitivity, specificity, etc., based on the predictions made by the classifier.
        - The user can specify which metrics to calculate by providing a list of metric names. By default, only accuracy is calculated.
        - Before evaluating, the method validates the shapes and types of the input test data.

        ---

        Parameters ⚒️:
        - `X_test (np.ndarray)`: The input feature array for testing.
        - `y_test (np.ndarray)`: The target output array for testing.
        - `metrics (List[str])`: Optional. A list of evaluation metric names to compute. Default is `["accuracy"]`. Available metrics are [
                "accuracy",
                "confusion_matrix",
                "confusion_metrics",
                "sensitivity",
                "specificity",
                "recall",
                "precision",
                "f1_score"
            ]

        ---

        Returns 📥:
        - A dictionary containing the calculated evaluation metrics.

        ---

        Raises ⛔:
        - `ArgumentError`: If any metric name provided is not in the list of available evaluation metric names.
        - `ValueError`: If the shapes of `X_test` and `y_test` are incompatible.
        - `TypeError`: If the types of `X_test` or `y_test` are not iterable.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.fit(X_train, y_train)

        # Load test data
        X_test = np.array([[4, 5], [5, 6]])
        y_test = np.array([0, 1])

        # Evaluate the classifier
        evaluation_results = knn.evaluate(X_test, y_test, metrics=["accuracy", "precision"])
        print(evaluation_results)
        ```
        """

        if any(metric not in self.evaluation_metric_names for metric in metrics):
            error_message = f"Choose correct metric names.\nGot: {metrics}.\nExpected: One of {self.evaluation_metric_names}"
            display_error(error_message=error_message, error_type=ArgumentError)

        self._validate_shapes(X_test, y_test, metrics="check_sample_count")
        self._validate_types(
            variable=X_test,
            variable_name="X_test",
            desired_type=Iterable,
            function=self.evaluate,
        )
        self._validate_types(
            variable=y_test,
            variable_name="y_test",
            desired_type=Iterable,
            function=self.evaluate,
        )

        # Ensure type consistency (np.ndarray)
        X_test = to_numpy(X_test)
        y_test = to_numpy(y_test)

        # Predict test labels
        y_real = y_test
        y_pred = self.predict(X_test)

        clfMetricsCalculator = self.clfMetrics(y_real=y_real, y_pred=y_pred)

        evaluation_results: dict = {}

        for metric in metrics:
            evaluation_function = self.evaluation_metric_mapping.get(metric)
            calculated_value: float | np.ndarray = evaluation_function(
                clfMetricsCalculator
            )

            evaluation_results.update({metric: calculated_value})

        return evaluation_results

    @_decorator_is_model_trained
    def test_the_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Test the trained KNN classifier on the provided test data.

        ---

        Description 📖
        - This method is equivalent to the `evaluate` method.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.fit(X_train, y_train)

        # Load test data
        X_test = np.array([[4, 5], [5, 6]])
        y_test = np.array([0, 1])

        # Test the classifier
        test_results = knn.test_the_model(X_test, y_test)
        print(test_results)
        ```
        """
        return self.evaluate(X_test=X_test, y_test=y_test)

    @_decorator_is_model_trained
    def predict(self, X_test: Iterable) -> np.ndarray:
        """
        Predict class labels for input data using the trained KNN classifier.

        ---

        Description 📖:
        - This method predicts class labels for input data using the trained KNN classifier.
        - It first validates the type of the input data and converts it to a numpy array if necessary.
        - The method then calls the internal `_predict` function for each sample in the input data to make predictions.
        - Finally, it converts the predictions to a numpy array and returns them.

        ---

        Parameters ⚒️:
        - `X_test (Iterable)`: The input data for which class labels are to be predicted.

        ---

        Returns 📤:
        - `np.ndarray`: An array containing the predicted class labels for each sample in `X_test`.

        ---

        Raises ⛔:
        - `TypeError`: If the type of `X_test` is not iterable.
        - `RuntimeError`: If the model is not trained.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.fit(X_train, y_train)

        # Make predictions
        X_test = np.array([[4, 5], [5, 6]])
        predictions = knn.predict(X_test)

        print(predictions)
        ```
        """
        self._validate_types(
            variable=X_test,
            variable_name="X_test",
            desired_type=Iterable,
            function=self.predict,
        )
        X_test = to_numpy(X_test)

        predictions = [self._predict(sample) for sample in X_test]
        predictions = to_numpy(predictions)
        return predictions

    def _predict(self, sample: np.ndarray) -> str:
        """
        Helper function to predict the class label for a single sample.

        ---

        Description 📖:
        - This internal function predicts the class label for a single sample using the trained KNN classifier.
        - It calculates the distances between the input sample and all training samples, selects the K nearest neighbors, and determines the most common class label among them.
        - The function performs encoding and decoding operations to handle categorical labels.
        - ❌ It is not recommended for direct use by the user, as its purpose is only internal to the KNN classifier.

        ---

        Parameters ⚒️:
        - `sample (np.ndarray)`: The input sample for which the class label is to be predicted.

        ---

        Returns 📤:
        - `str`: The predicted class label for the input sample.

        ---

        Raises ⛔:
        - `TypeError`: If the type of `sample` is not iterable.

        ---

        Example 🎯:
        ```
        # Predict the class label for a single sample
        sample = np.array([4, 5])
        prediction = _predict(sample)
        print(prediction)
        ```
        """

        self._validate_types(
            variable=sample,
            variable_name="(HELPER FUNCTION) sample",
            desired_type=Iterable,
            function=self._predict,
        )
        sample = to_numpy(sample)

        distances = self._get_euclidean_distance(sample, self.X_train)

        k_indexes = np.argsort(distances)[: self.K]

        k_labels = self.y_train[k_indexes]
        print(k_labels)
        if not self.regression_mode:

            k_labels = self._encode(k_labels)

            label_count = np.bincount(k_labels)
            common_label = np.argmax(label_count)

            prediction = common_label
            prediction = self._decode(common_label.flatten())[0]
        else:
            prediction = np.mean(k_labels)

        return prediction

    def _get_euclidean_distance(
        self, array_1: np.ndarray, array_2: np.ndarray
    ) -> np.ndarray:
        """
        Internal function to calculate the Euclidean distance between two arrays.

        ---

        Description 📖:
        - This internal function calculates the Euclidean distance between two arrays.
        - It is used as a distance metric in the KNN algorithm to measure the similarity between data points.
        - The function validates the types of the input arrays and converts them to numpy arrays if necessary.

        ---

        Parameters ⚒️:
        - `array_1 (np.ndarray)`: The first input array.
        - `array_2 (np.ndarray)`: The second input array.

        ---

        Returns 📤:
        - `np.ndarray`: The Euclidean distance between the input arrays.

        ---

        Raises ⛔:
        - `TypeError`: If the types of `array_1` or `array_2` are not iterable.

        ---

        Example 🎯:
        ```
        # Calculate Euclidean distance between two arrays
        array1 = np.array([1, 2, 3])
        array2 = np.array([4, 5, 6])
        distance = _get_euclidean_distance(array1, array2)
        print(distance)
        ```
        """

        self._validate_types(
            variable=array_1,
            variable_name="array_1",
            desired_type=Iterable,
            function=self._get_euclidean_distance,
        )
        self._validate_types(
            variable=array_2,
            variable_name="array_2",
            desired_type=Iterable,
            function=self._get_euclidean_distance,
        )

        array_1 = to_numpy(array_1)
        array_2 = to_numpy(array_2)

        euclidean_distance = np.sqrt(np.sum((array_1 - array_2) ** 2, axis=-1))
        return euclidean_distance

    @_decorator_is_model_trained
    def plot_data(
        self,
        X_test=None,
        y_test=None,
        metrics: List[Literal["training", "testing", "decision_boundary"]] = [
            "training",
            "testing",
        ],
        figsize: Tuple[int, int] = (8, 6),
        save_plot: bool = False,
        filename: str = "MyPlot.png",
    ):
        """
        Plot the data points and optionally the decision boundary.

        ---

        Description 📖:
        - This method plots the data points along with the decision boundary if specified.
        - It supports different metrics for plotting, including training data points, testing data points, and the decision boundary.
        - The method validates the provided metrics and raises an error if any invalid metric is given.
        - The decision boundary is computed using the KNN classifier and is displayed if specified in the metrics.
        - The plot can be saved as an image file if `save_plot` is set to True.

        ---

        Parameters ⚒️:
        - `X_test (np.ndarray)`: The input features of the testing data.
        - `y_test (np.ndarray)`: The output labels of the testing data.
        - `metrics (List[str])`: A list of metrics to plot, including 'training', 'testing', and 'decision_boundary'.
        - `figsize (Tuple[int, int])`: The size of the figure (width, height) in inches.
        - `save_plot (bool)`: A flag indicating whether to save the plot as an image file.
        - `filename (str)`: The filename for the saved plot.

        ---

        Returns 📥:
        - `plt`: The matplotlib.pyplot object containing the plot.

        ---

        Raises ⛔:
        - `ArgumentError`: If any invalid metric is provided in the `metrics` parameter.

        ---

        Example 🎯:
        ```
        # Create an instance of KNN classifier
        knn = KNN(K=3, verbose=1)

        # Load training data
        X_train = np.array([[1, 2], [2, 3], [3, 4]])
        y_train = np.array([0, 1, 0])

        # Train the classifier
        knn.fit(X_train, y_train)

        # Plot the decision boundary and training data
        my_plt = knn.plot_data(metrics=["training", "decision_boundary"])
        my_plt.show()
        ```
        """
        available_metrics = ["training", "testing", "decision_boundary"]
        if any(metric not in available_metrics for metric in metrics):
            error_message = f"Choose correct metric names.\nGot: {metrics}.\nExpected: One of {available_metrics}"
            display_error(error_message=error_message, error_type=ArgumentError)

        X_train = self.X_train
        y_train = self.y_train

        plt.figure(figsize=figsize)

        plt.title("Visualization of Data Points")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        if "decision_boundary" in metrics:
            info_message = "Defining decision boundary......Please wait."
            self._display_info_message(info_message)

            clf = self
            X = self.X_train
            y = self.y_train
            y = self._encode(self.y_train)

            h = 0.5  # Larger step size for faster plotting

            # Create meshgrid of feature values
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            # Predict the class for each point in the meshgrid
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = self._encode(Z)
            Z = Z.reshape(xx.shape)

            # Plot the decision boundary
            plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

            plt.title("Decision Boundary")

        # Plot training data
        if "training" in metrics:

            # Plot each class separately
            for label in np.unique(y_train):
                # Select data points with the current label
                X_class = X_train[y_train == label]

                # Plot the 2D data points (x,y) with a different color for each class
                plt.scatter(
                    X_class[:, 0],
                    X_class[:, 1],
                    label=f"Training - Class `{label}`",
                    marker="o",
                )

            plt.legend()

        # Plot test data
        if "testing" in metrics:

            if X_test is not None and y_test is not None:
                for label in np.unique(y_test):
                    X_class = X_test[y_test == label]

                    plt.scatter(
                        X_class[:, 0],
                        X_class[:, 1],
                        label=f"Test - Class {label}",
                        marker="x",
                    )
                plt.legend()
            else:
                warning_message = f"To plot `testing` metrics, parameters `X_test` and `y_test` shouldn't to equal `None`."
                display_warning(warning_message=warning_message)

        if save_plot:
            plt.savefig(filename)

        return plt

    def generate_dataset(
        # Generates dataset with 2d points (x,y)
        self,
        n_samples=15,
        n_classes: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a synthetic dataset for testing the KNN classifier.

        ---

        Description 📖:
        - This method generates a synthetic dataset with 2D points (x,y) for training and testing the KNN classifier.
        - The dataset consists of multiple classes, and the number of samples and classes can be specified.
        - Each class is represented by a set of points with random coordinates within a certain range.

        ---

        Parameters ⚒️:
        - `samples (int)`: The total number of samples to generate for all classes.
        - `n_classes (int)`: The number of classes in the dataset.

        ---

        Returns 📤:
        - A tuple containing the training and testing data:
            - `X_train (np.ndarray)`: The training feature array.
            - `y_train (np.ndarray)`: The training target array.
            - `X_test (np.ndarray)`: The testing feature array.
            - `y_test (np.ndarray)`: The testing target array.

        ---

        Example 🎯:
        ```
        # Generate synthetic dataset with 15 samples and 3 classes
        X_train, y_train, X_test, y_test = generate_dataset(samples=15, n_classes=3)
        ```
        """

        if not self.regression_mode:

            X_train = np.concatenate(
                (
                    [
                        np.random.randint(
                            low=1 * (i * 50),
                            high=50 * (i + 1),
                            size=(n_samples // n_classes, 2),
                        )
                        for i in range(n_classes)
                    ]
                )
            )
            X_test = np.concatenate(
                (
                    [
                        np.random.randint(
                            low=1 * (i * 50),
                            high=50 * (i + 1),
                            size=(n_samples // n_classes, 2),
                        )
                        for i in range(n_classes)
                    ]
                )
            )

            y_train = y_test = np.concatenate(
                (
                    [
                        np.repeat("MyClass_" + str(i), n_samples // n_classes)
                        for i in range(n_classes)
                    ]
                ),
            )
        else:
            k, b = np.random.randint(low=1, high=15, size=(2,))
            print(k, b)
            X_train = np.random.randint(low=1, high=100, size=(n_samples,))
            X_test = np.random.randint(low=1, high=100, size=(n_samples,))

            y_train = X_train * k + b
            y_test = X_test * k + b
        X_train = X_train.reshape((-1, 1))
        X_test = X_test.reshape((-1, 1))
        X_train, y_train = self._simultaneous_shuffle(X_train, y_train)
        X_test, y_test = self._simultaneous_shuffle(X_test, y_test)

        return X_train, y_train, X_test, y_test

    @_decorator_is_model_trained
    def save_model(self, filename="KNN_model.pkl") -> None:
        """
        Save the trained model to a file.

        ---

        Description 📖:
        - This method saves the trained KNN classifier instance to a file using pickle serialization.
        - The model must be trained before calling this method.
        - The default filename is "KNN_model.pkl", but a custom filename can be provided.

        ---

        Parameters ⚒️:
        - `filename (str)`: The filename to save the model to. Default is "KNN_model.pkl".

        ---

        Returns 📥:
        - None

        ---

        Raises ⛔:
        - `RuntimeError`: If the model is not trained when calling this method.

        ---

        Example 🎯:
        ```
        # Save the trained model to a file
        knn.save_model(filename="my_model.pkl")
        ```
        """
        save_instance(self, filename=filename)

    def load_model(self, filename="KNN_model.pkl") -> Any:
        """
        Load a trained model from a file.

        ---

        Description 📖:
        - This method loads a trained KNN classifier instance from a file using pickle deserialization.
        - The filename to load the model from must be provided.
        - The loaded model can be assigned to a variable for further use.

        ---

        Parameters ⚒️:
        - `filename (str)`: The filename to load the model from. Default is "KNN_model.pkl".

        ---

        Returns 📤:
        - `Any`: The loaded KNN classifier instance.

        ---

        Example 🎯:
        ```
        # Load a trained model from a file
        loaded_model = knn.load_model(filename="my_model.pkl")
        ```
        """
        return load_instance(filename=filename)

    def _validate_shapes(
        self,
        array_1: np.ndarray,
        array_2: np.ndarray,
        metrics: Literal["check_sample_count", "check_full_shape"],
        ignore_empty_arrays: bool = False,
        print_function=None,
    ) -> None:
        """
        Internal function to validate the shapes of two numpy arrays.

        Example 🎯:
        ```
        # Validate the shapes of two numpy arrays
        array1 = np.array([[1, 2], [3, 4]])
        array2 = np.array([5, 6])
        _validate_shapes(array1, array2, metrics="check_sample_count")
        ```
        """
        if print_function is None and self.verbose == 2:
            print_function = self._display_guide_message

        validate_shapes(
            array_1=array_1,
            array_2=array_2,
            metrics=metrics,
            ignore_empty_arrays=ignore_empty_arrays,
            print_function=print_function,
        )

    def _validate_types(
        self,
        variable: Any,
        variable_name: str,
        desired_type: type | List,
        function: Callable = None,
        print_function=None,
    ) -> None:
        """
        Internal function to validate the type of a variable.

        Example 🎯:
        ```
        # Validate the type of a variable
        variable = "example"
        _validate_types(variable, "variable_name", str)
        ```
        """
        if print_function is None and self.verbose == 2:
            print_function = self._display_guide_message

        validate_types(
            variable=variable,
            variable_name=variable_name,
            desired_type=desired_type,
            function=function,
            print_function=print_function,
        )

    def _simultaneous_shuffle(
        self, array_1: np.ndarray, array_2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal function to simultaneously shuffle two numpy arrays.

        Example 🎯:
        ```
        # Simultaneously shuffle two numpy arrays
        array1 = np.array([1, 2, 3])
        array2 = np.array(['a', 'b', 'c'])
        shuffled_array1, shuffled_array2 = _simultaneous_shuffle(array1, array2)
        print(shuffled_array1)
        print(shuffled_array2)
        ```
        """
        return simultaneous_shuffle(array_1=array_1, array_2=array_2)
