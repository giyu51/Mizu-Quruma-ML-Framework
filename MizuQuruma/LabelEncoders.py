from typing import Iterable, Callable, Any, Dict, NoReturn
from .message_formatter import format_message
from .model_selection import save_instance, load_instance
import numpy as np
from .internal_methods import to_numpy


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
            - `TypeError`: If the `vocabulary` parameter is not an iterable object.
            - `ValueError`: If the provided vocabulary is empty.
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

    def _decorator_is_model_trained(
        _function: Callable[..., Any]
    ) -> Callable[..., Any]:
        """
        Decorator to ensure that the encoder is fitted before calling a method.

        Description 📖:
        - This decorator checks whether the encoder is fitted before allowing
        the decorated method to be called.
        - If the encoder is not fitted, it raises a RuntimeError indicating that
        the encoder must be fitted first using the
        `fit()` method.

        ---

        Parameters 🛠️:
            - `_function` (Callable[..., Any]): The method to be decorated.

        ---

        Returns 📤:
            - `callable`: The wrapper function.

        ---

        Raises ⛔:
            - `RuntimeError`: If the encoder is not fitted.
        """

        def wrapper(self, *args, **kwargs):
            if not self.encoder_is_fit:
                message = f"To use `{_function.__name__}()`, the encoder should be fitted using `fit()`"
                message = format_message(msg=message, msg_type="error")
                raise RuntimeError(message)
            return _function(self, *args, **kwargs)

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

    @_decorator_is_model_trained
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

    @_decorator_is_model_trained
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

    @_decorator_is_model_trained
    def save_encoder(self, filename: str = "StringToInt_encoder.pkl") -> None:
        """
        Save the encoder instance to a file using pickle serialization.

        Parameters 🛠️:
            - `filename` (str): The filename to save the encoder instance to. Default is 'StringToInt_encoder.pkl'.

        ---

        Returns 📥:
            - None


        Raises ⛔:
            - `TypeError`: If the instance cannot be serialized using pickle.
            - `FileNotFoundError`: If the specified file or directory does not exist.

        ---

        Example💡:
            >>> # Save the encoder instance to a file named 'encoder.pkl'
            encoder = StringToIntEncoder()
            encoder.fit(vocabulary)
            encoder.save_encoder("encoder.pkl")
        """

        save_instance(self, filename=filename)

    def load_encoder(self, filename: str = "StringToInt_encoder.pkl") -> Any:
        """
        Load the encoder instance from a file using pickle deserialization.

        Args:
            filename (str): The filename to load the encoder instance from. Default is 'StringToInt_encoder.pkl'.

        Returns:
            Any: The loaded encoder instance.

        Example:
            >>> # Load the encoder instance from a file named 'encoder.pkl'
            >>> encoder = StringToIntEncoder()
            >>> encoder = encoder.load_encoder("encoder.pkl")

        Raises:
            FileNotFoundError: If the specified file or directory does not exist.
        """
        return load_instance(filename=filename)


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
