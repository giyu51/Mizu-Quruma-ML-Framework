from typing import Iterable, Callable, Any, Dict, NoReturn
from .message_formatter import format_message
from .model_selection import save_instance
import numpy as np


class StringToIntEncoder:
    def __init__(self) -> None:
        self.label_mapping = {}
        self.encoder_is_fit = False

    def fit(self, vocabulary: Iterable[str]) -> Dict[str, int]:
        """
        Fit the encoder to the given vocabulary, mapping each unique label to an integer index.

        Args🛠️:
            - `vocabulary` (Iterable[str]): The vocabulary containing unique labels to be encoded.

        ---

        Returns📤:
            - `Dict[str, int]`: A dictionary mapping each unique label to its corresponding integer index.

        ---

        Raises ⛔:
            - `ValueError`: If the encoder has already been fitted.
        """
        if self.encoder_is_fit:
            raise ValueError("Encoder has already been fitted.")
        if not isinstance(vocabulary, Iterable):
            message = "The `vocabulary` parameter must be an iterable object."
            message = format_message(msg=message, msg_type="error")
            raise TypeError(message)
        if len(vocabulary) == 0:
            message = "Provided vocabulary cannot be empty. Please provide non-empty vocabulary for fitting the encoder."
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

        unique_labels = np.unique(vocabulary)

        for idx, label in enumerate(unique_labels):
            if label not in self.label_mapping:
                self.label_mapping[label] = idx

        self.reverse_label_mapping = {
            value: key for key, value in self.label_mapping.items()
        }

        self.encoder_is_fit = True
        return self.label_mapping

    def reset(self) -> None:
        """
        Reset the state of the StringToIntEncoder instance.

        Description📝:
            - This method resets the state of the StringToIntEncoder instance by clearing
            the label mappings and setting the encoder's fit status to False. This can
            be useful when you want to clear the encoder's state and start fresh,
            for example, when you want to fit the encoder with a new vocabulary.

        ---

        Example💡:
            >>> encoder = StringToIntEncoder()
            >>> encoder.fit(['apple', 'banana'])
            >>> encoder.reset()
            >>> encoder.fit(['orange', 'kiwi'])

        ---

        Note⚠️:
            After calling this method, the encoder will no longer retain any information
            about previously fitted vocabularies or encoded labels.
        """
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        self.encoder_is_fit = False

    def to_numpy(self, _arr: Iterable) -> np.ndarray:
        """
        Convert an iterable to a numpy array.

        Description 📝:
            - This method automatically handles conversion of iterable data (e.g., lists, tuples)
                to numpy arrays, making it easy to work with various data formats.

            - If the input is already a numpy array, the method returns the input array itself.

        ---

        Args 🛠️:
            - `_arr` (Iterable): The iterable to be converted.

        ---

        Returns 📤:
            - `np.ndarray`: The numpy array resulting from the conversion, or the input array if it is already a numpy array.

        ---

        Raises ⛔:
            - `ValueError`: If the conversion fails due to invalid input data.
            - `RuntimeError`: If an unexpected error occurs during conversion.
        """
        try:
            numpy_array = np.array(_arr) if not isinstance(_arr, np.ndarray) else _arr
            return numpy_array
        except ValueError as ve:
            message = f"Failed to convert input to numpy array:\n {ve}"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)
        except Exception as e:
            message = f"An unexpected error occurred:\n {e}"
            message = format_message(msg=message, msg_type="error")

            raise RuntimeError(message)

    def ensure_encoder_is_fit(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorator to ensure that the encoder is fitted before calling a method.

        Description 📝:

            - This decorator checks whether the encoder is fitted before allowing
        the decorated method to be called. If the encoder is not fitted, it raises
        a RuntimeError indicating that the encoder must be fitted first using the
        `fit()` method.

        ---

        Args 🛠️:
            - `func` (callable): The method to be decorated.

        ---

        Returns 📤:
            - `callable`: The wrapper function.

        ---

        Raises ⛔:
            - `RuntimeError`: If the encoder is not fitted.
        """

        def wrapper(self, *args, **kwargs):
            if not self.encoder_is_fit:
                message = f"To use `{func.__name__}()`, the encoder should be fitted using `fit()`"
                message = format_message(msg=message, msg_type="error")
                raise RuntimeError(message)
            return func(self, *args, **kwargs)

        return wrapper

    def _check_unknown_symbols(
        self, X: np.array, test_elements: Iterable
    ) -> None | NoReturn:
        """
        Check for unknown symbols in the input array X.
        Raise a ValueError if any unknown symbols are found.

        Args 🛠️:
            - `X` (np.array): The input array to check for unknown symbols.
            - `test_elements` (Iterable): The elements to test for membership in the input array.

        ---

        Raises ⛔:
            - `ValueError`: If any unknown symbols are found in the input array.
        """
        unknown_mask = np.isin(X, test_elements, invert=True)
        if np.any(unknown_mask):
            unknown_values = X[unknown_mask]

            message = f"Found value(s) that are not in the vocabulary:\n\u2192 {unknown_values}"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

    @ensure_encoder_is_fit
    def encode(
        self, X: np.array, replace_unknown=True, unknown_symbol: int = -1
    ) -> np.ndarray:
        """
        Encode the input array using the fitted label encoder.

        Args 🛠️:
            - `X` (np.array): The input array to be encoded.
            - `replace_unknown` (bool): Whether to replace unknown labels with the specified `unknown_symbol`.
            - `unknown_symbol` (int): The value to use for unknown labels if `replace_unknown` is True.

        ---

        Returns 📤:
            - `np.ndarray`: The encoded array.

        ---

        Raises ⛔:
            - `ValueError`: If any unknown labels are encountered and `replace_unknown` is False.
            - `RuntimeError`: If the encoder is not fitted.
        """
        X = self.to_numpy(X)
        if not replace_unknown:
            self._check_unknown_symbols(X, list(self.label_mapping.keys()))

        encoded_list = np.array(
            [self.label_mapping.get(label, unknown_symbol) for label in X]
        )
        return encoded_list

    @ensure_encoder_is_fit
    def decode(
        self, X: np.array, replace_unknown=True, unknown_symbol: str = "?"
    ) -> np.ndarray:
        """
        Decode the input array using the fitted label decoder.

        Args 🛠️:
            - `X` (np.array): The input array to be decoded.
            - `replace_unknown` (bool): Whether to replace unknown labels with the specified `unknown_symbol`.
            - `unknown_symbol` (str): The symbol to use for unknown labels if `replace_unknown` is True.

        ---

        Returns 📤:
            - `np.ndarray`: The decoded array.

        ---

        Raises ⛔:
            - `ValueError`: If any unknown labels are encountered and `replace_unknown` is False.
            - `RuntimeError`: If the encoder is not fitted.
        """
        X = self.to_numpy(X)
        if not replace_unknown:
            self._check_unknown_symbols(X, list(self.reverse_label_mapping.keys()))

        decoded_list = np.array(
            [self.reverse_label_mapping.get(num, unknown_symbol) for num in X]
        )
        return decoded_list

    @ensure_encoder_is_fit
    def save(self, filename: str = "StringToInt_encoder.pkl") -> None:
        """
        Save the encoder instance to a file using pickle serialization.

        Args🛠️:
            - `filename` (str): The filename to save the encoder instance to. Default is 'StringToInt_encoder.pkl'.

        ---

        Example💡:
            >>> # Save the encoder instance to a file named 'encoder.pkl'
            encoder = StringToIntEncoder()
            encoder.fit(vocabulary)
            encoder.save("encoder.pkl")

        ---

        Raises ⛔:
            - `TypeError`: If the instance cannot be serialized using pickle.
            - `FileNotFoundError`: If the specified file or directory does not exist.
        """

        save_instance(self, filename=filename)


"""
TESTS ARE DONE

Sure! Here's the list with small descriptions for each test case:

1. **Test Basic Fit and Transform**: Fit the encoder with a simple vocabulary and then encode a list of labels to verify the encoding.
   
2. **Test Encoding Unknown Labels**: Test encoding with labels that were not present in the vocabulary during fitting. Ensure that it handles unknown labels appropriately based on the `replace_unknown` parameter.
   
3. **Test Decoding Unknown Labels**: Similarly, decode labels that were not present in the original vocabulary. Ensure that it handles unknown labels correctly based on the `replace_unknown` parameter.
   
4. **Test Fit Multiple Times**: Attempt to fit the encoder multiple times with different vocabularies. Ensure that it raises an error when trying to fit the encoder again without resetting it.
   
5. **Test Save and Load**: Fit the encoder, save it to a file, and then load it back. Verify that the loaded encoder produces the same results as the original one.
   
6. **Test Edge Cases**: Test with edge cases such as empty vocabularies, empty input arrays, or arrays with a single label.
   
7. **Test Type Checking**: Test with inputs of different types (e.g., lists, tuples, numpy arrays) to ensure that type checking works correctly.
   
8. **Test Large Dataset**: Test with a large dataset to ensure that it can handle the computation efficiently without running into memory or performance issues.

These descriptions give a brief overview of what each test case is intended to verify or validate in the `StringToIntEncoder` class.
"""