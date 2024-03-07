from typing import Iterable, Callable, Any, Dict, NoReturn

import numpy as np

from .model_selection import save_instance, load_instance
from .internal_methods import to_numpy, display_error


class StringToIntEncoder:
    def __init__(self) -> None:
        """
        Class for encoding and decoding strings to integers.

        ---

        Description 📖:
        - This class provides methods for encoding and decoding strings to integers, useful for converting categorical variables into numerical format for machine learning models.
        - After initializing an instance of StringToIntEncoder, you need to fit the encoder to a vocabulary using the `fit()` method before encoding or decoding strings.

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
        # Create an instance of StringToIntEncoder
        encoder = StringToIntEncoder()

        # Fit the encoder to a vocabulary
        vocabulary = ['apple', 'banana', 'orange']
        encoder.fit(vocabulary)

        # Encode and decode strings
        encoded_array = encoder.encode(['apple', 'banana', 'orange'])
        decoded_array = encoder.decode(encoded_array)

        print(encoded_array)
        print(decoded_array)
        ```
        """
        self.label_mapping = {}
        self.encoder_is_fit = False

    def fit(self, vocabulary: Iterable[str]) -> Dict[str, int]:
        """
        Fit the encoder to the given vocabulary, mapping each unique label to an integer index.

        ---

        Description 📖:
        - This method fits the encoder to the provided vocabulary, mapping each unique label to an integer index.

        ---

        Parameters ⚒️:
        - `vocabulary` (Iterable[str]): The vocabulary containing unique labels to be encoded.

        ---

        Returns 📥:
        - `Dict[str, int]`: A dictionary mapping each unique label to its corresponding integer index.

        ---

        Raises ⛔:
        - `ValueError`: If the encoder has already been fitted.
        - `TypeError`: If the `vocabulary` parameter is not an iterable object.
        - `ValueError`: If the provided vocabulary is empty.

        ---

        Example 🎯:
        ```
        # Create an instance of StringToIntEncoder
        encoder = StringToIntEncoder()

        # Fit the encoder to a vocabulary
        vocabulary = ['apple', 'banana', 'orange']
        label_mapping = encoder.fit(vocabulary)

        print(label_mapping)
        ```
        """

        if self.encoder_is_fit:
            error_message = "Encoder has already been fitted."
            display_error(error_message=error_message, error_type=ValueError)
        if not isinstance(vocabulary, Iterable):
            error_message = "The `vocabulary` parameter must be an iterable object."
            display_error(error_message=error_message, error_type=TypeError)
        if len(vocabulary) == 0:
            error_message = "Provided vocabulary cannot be empty. Please provide non-empty vocabulary for fitting the encoder."
            display_error(error_message=error_message, error_type=ValueError)

        unique_labels = np.unique(vocabulary)

        for idx, label in enumerate(unique_labels):
            if label not in self.label_mapping:
                self.label_mapping[label] = idx

        self.reverse_label_mapping = {
            value: key for key, value in self.label_mapping.items()
        }

        self.encoder_is_fit = True
        return self.label_mapping

    def reset_encoder(self) -> None:
        """
        Reset the state of the StringToIntEncoder instance.

        Description📝:
        - This method resets the state of the StringToIntEncoder instance by clearing
        the label mappings and setting the encoder's fit status to False. This can
        be useful when you want to clear the encoder's state and start fresh,
        for example, when you want to fit the encoder with a new vocabulary.

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

        Example💡:
        ```
        # Create an instance of StringToIntEncoder
        encoder = StringToIntEncoder()

        # Fit the encoder to a vocabulary
        vocabulary = ['apple', 'banana']
        encoder.fit(vocabulary)

        # Reset the encoder
        encoder.reset_encoder()

        # Fit the encoder with a new vocabulary
        encoder.fit(['orange', 'kiwi'])
        ```

        ---

        Note⚠️:
        After calling this method, the encoder will no longer retain any information
        about previously fitted vocabularies or encoded labels.
        """
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        self.encoder_is_fit = False

    def _decorator_is_model_trained(
        _function: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        Decorator to ensure that the encoder is fitted before calling a method.

        ---

        Description 📖:
        - This decorator checks whether the encoder is fitted before allowing the decorated method to be called.
        - If the encoder is not fitted, it raises a RuntimeError indicating that the encoder must be fitted first using the `fit()` method.

        ---

        Parameters ⚒️:
        - `_function` (Callable[..., Any]): The method to be decorated.

        ---

        Returns 📤:
        - `callable`: The wrapper function.

        ---

        Raises ⛔:
        - `RuntimeError`: If the encoder is not fitted.

        ---

        Example 🎯:
        ```python
        @_decorator_is_model_trained
        def some_method(self, *args, **kwargs):
            # Method implementation
        ```
        """

        def wrapper(self, *args, **kwargs):
            if not self.encoder_is_fit:
                error_message = f"To use `{_function.__name__}()`, the encoder should be fitted using `fit()`"
                display_error(error_message=error_message, error_type=RuntimeError)
            return _function(self, *args, **kwargs)

        return wrapper

    def _check_unknown_symbols(
        self, X: np.array, test_elements: Iterable
    ) -> None | NoReturn:
        """
        ⚠️ INTERNAL METHOD

        Check for unknown symbols in the input array X.

        ---

        Description 📖:
        - This method checks for unknown symbols in the input array `X`. It raises a `ValueError` if any unknown symbols are found.

        ---

        Parameters ⚒️:
        - `X (np.array)`: The input array to check for unknown symbols.
        - `test_elements (Iterable)`: The elements to test for membership in the input array.

        ---

        Raises ⛔:
        - `ValueError`: If any unknown symbols are found in the input array.

        ---

        Example 🎯:
        ```
        # Define test data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        test_elements = [2, 4, 6]

        # Check for unknown symbols
        _check_unknown_symbols(X, test_elements)
        ```
        """
        unknown_mask = np.isin(X, test_elements, invert=True)
        if np.any(unknown_mask):
            unknown_values = X[unknown_mask]

            error_message = f"Found value(s) that are not in the vocabulary:\n\u2192 {unknown_values}"
            display_error(error_message=error_message, error_type=ValueError)

    @_decorator_is_model_trained
    def encode(
        self, X: np.array, replace_unknown=True, unknown_symbol: int = -1
    ) -> np.ndarray:
        """
        Encode the input array using the fitted label encoder.

        ---

        Description 📖:
        - This method encodes the input array `X` using the fitted label encoder.
        - Optionally, it can replace unknown labels with the specified `unknown_symbol`.
        - If any unknown labels are encountered and `replace_unknown` is False, a ValueError is raised.

        ---

        Parameters ⚒️:
        - `X (np.array)`: The input array to be encoded.
        - `replace_unknown (bool)`: Whether to replace unknown labels with the specified `unknown_symbol`. By defualt is `True`.
        - `unknown_symbol (int)`: The value to use for unknown labels if `replace_unknown` is True. By default is `-1`

        ---

        Returns 📤:
        - `np.ndarray`: The encoded array.

        ---

        Raises ⛔:
        - `ValueError`: If any unknown labels are encountered and `replace_unknown` is False.
        - `RuntimeError`: If the encoder is not fitted.

        ---

        Example 🎯:
        ```
        # Define input array
        X = np.array(['apple', 'banana', 'orange'])

        # Encode the input array
        encoded_array = knn.encode(X, replace_unknown=True, unknown_symbol=-1)
        print(encoded_array)
        ```
        """
        X = to_numpy(X)
        if not replace_unknown:
            self._check_unknown_symbols(X, list(self.label_mapping.keys()))

        encoded_list = np.array(
            [self.label_mapping.get(label, unknown_symbol) for label in X]
        )
        return encoded_list

    @_decorator_is_model_trained
    def decode(
        self, X: np.array, replace_unknown=True, unknown_symbol: str = "?"
    ) -> np.ndarray:
        """
        Decode the input array using the fitted label decoder.

        ---

        Description 📖:
        - This method decodes the input array `X` using the fitted label decoder.
        - Optionally, it can replace unknown labels with the specified `unknown_symbol`.
        - If any unknown labels are encountered and `replace_unknown` is False, a ValueError is raised.

        ---

        Parameters ⚒️:
        - `X (np.array)`: The input array to be decoded.
        - `replace_unknown (bool)`: Whether to replace unknown labels with the specified `unknown_symbol`. By default is `True`
        - `unknown_symbol (str)`: The symbol to use for unknown labels if `replace_unknown` is True. By default is `"?"`

        ---

        Returns 📤:
        - `np.ndarray`: The decoded array.

        ---

        Raises ⛔:
        - `ValueError`: If any unknown labels are encountered and `replace_unknown` is False.
        - `RuntimeError`: If the encoder is not fitted.

        ---

        Example 🎯:
        ```
        # Define input array
        X = np.array(['apple', 'banana', 'orange'])

        # Encode input array
        encoded_X = knn.encode(X)

        # Decode the input array
        decoded_array = knn.decode(encoded_X, replace_unknown=True, unknown_symbol='?')
        print(decoded_array)
        ```
        """
        X = to_numpy(X)
        if not replace_unknown:
            self._check_unknown_symbols(X, list(self.reverse_label_mapping.keys()))

        decoded_list = np.array(
            [self.reverse_label_mapping.get(num, unknown_symbol) for num in X]
        )
        return decoded_list

    @_decorator_is_model_trained
    def save_encoder(self, filename: str = "StringToInt_encoder.pkl") -> None:
        """
        Save the encoder instance to a file using pickle serialization.

        ---

        Description 📖:
        - This method saves the encoder instance to a file using pickle serialization.
        - The encoder instance is saved to the specified `filename` using the default filename 'StringToInt_encoder.pkl' if not provided.

        ---

        Parameters ⚒️:
        - `filename (str)`: The filename to save the encoder instance to. Default is 'StringToInt_encoder.pkl'.

        ---

        Returns 📥:
        - None

        ---

        Raises ⛔:
        - `TypeError`: If the instance cannot be serialized using pickle.
        - `FileNotFoundError`: If the specified file or directory does not exist.

        ---

        Example 🎯:
        ```
        # Save the encoder instance to a file named 'encoder.pkl'
        encoder = StringToIntEncoder()
        encoder.fit(vocabulary)
        encoder.save_encoder("encoder.pkl")
        ```
        """

        save_instance(self, filename=filename)

    def load_encoder(self, filename: str = "StringToInt_encoder.pkl") -> Any:
        """
        Load the encoder instance from a file using pickle deserialization.

        ---

        Description 📖:
        - This method loads the encoder instance from a file using pickle deserialization.
        - The encoder instance is loaded from the specified `filename` using the default filename 'StringToInt_encoder.pkl' if not provided.

        ---

        Parameters ⚒️:
        - `filename (str)`: The filename to load the encoder instance from. Default is 'StringToInt_encoder.pkl'.

        ---

        Returns 📥:
        - Any: The loaded encoder instance.

        ---

        Raises ⛔:
        - `FileNotFoundError`: If the specified file or directory does not exist.

        ---

        Example 🎯:
        ```
        # Load the encoder instance from a file named 'encoder.pkl'
        encoder = StringToIntEncoder()
        encoder = encoder.load_encoder("encoder.pkl")
        ```
        """
        return load_instance(filename=filename)


"""
TESTS ARE DONE

1. **Test Basic Fit and Transform**: Fit the encoder with a simple vocabulary and then encode a list of labels to verify the encoding.
   
2. **Test Encoding Unknown Labels**: Test encoding with labels that were not present in the vocabulary during fitting. Ensure that it handles unknown labels appropriately based on the `replace_unknown` parameter.
   
3. **Test Decoding Unknown Labels**: Similarly, decode labels that were not present in the original vocabulary. Ensure that it handles unknown labels correctly based on the `replace_unknown` parameter.
   
4. **Test Fit Multiple Times**: Attempt to fit the encoder multiple times with different vocabularies. Ensure that it raises an error when trying to fit the encoder again without resetting it.
   
5. **Test Save and Load**: Fit the encoder, save it to a file, and then load it back. Verify that the loaded encoder produces the same results as the original one.
   
6. **Test Edge Cases**: Test with edge cases such as empty vocabularies, empty input arrays, or arrays with a single label.
   
7. **Test Type Checking**: Test with inputs of different types (e.g., lists, tuples, numpy arrays) to ensure that type checking works correctly.
   
8. **Test Large Dataset**: Test with a large dataset to ensure that it can handle the computation efficiently without running into memory or performance issues.

"""
