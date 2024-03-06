from typing import Any, Iterable, List, Literal, get_args, Callable
import numpy as np
from icecream import ic
from .message_formatter import format_message


def validate_shapes(
    array_1: np.ndarray,
    array_2: np.ndarray,
    metrics: Literal["check_sample_count", "check_full_shape"],
    print_function=None,
    ignore_empty_arrays: bool = False,
) -> None:

    if print_function is None:
        print_function = noop

    validate_types(
        variable=array_1,
        variable_name=f"array_1",
        desired_type=Iterable,
        function=validate_shapes,
    )
    validate_types(
        variable=array_2,
        variable_name=f"array_2",
        desired_type=Iterable,
        function=validate_shapes,
    )
    validate_types(
        variable=metrics,
        variable_name=f"metrics",
        desired_type=Literal["check_sample_count", "check_full_shape"],
        function=validate_shapes,
    )
    validate_types(
        variable=ignore_empty_arrays,
        variable_name=f"ignore_empty_arrays",
        desired_type=bool,
        function=validate_shapes,
    )

    # Ensure inputs are numpy arrays
    X = to_numpy(array_1)
    y = to_numpy(array_2)

    # Get the number of samples
    X_samples, y_samples = X.shape[0], y.shape[0]

    if X.size == 0 or y.size == 0:
        if ignore_empty_arrays:
            message = "⚠️ IGNORING: Input array(s) are empty. To raise an error in this case, make sure `ignore_empty_arrays` flag is set to `False`"
            print_function(message)
        else:
            message = "❌ Input array(s) are empty. All arrays must contain samples. Got:\nArray 1 `shape`: {X.shape}\nArray 2 `shape`: {y.shape}"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

    if metrics == "check_sample_count":
        if X_samples == y_samples:
            message = f"✅ SAMPLE COUNT CHECK Validation is Approved.\nNumber of samples are consistent in both Array 1 and Array 2.\nArray 1 `shape`: {X.shape} \t\u2192 {X_samples} samples\nArray 2 `shape`: {y.shape} \t\u2192 {y_samples} samples"
            print_function(message)
        else:
            message = f"❌ Inconsistent number of samples between Array 1 and Array 2. Number of samples should be consistent both in Array 1 and Array 2. Got:\nArray 1 `shape`: {X.shape} \t\u2192 {X_samples} samples\nArray 2 `shape`: {y.shape} \t\u2192 {y_samples} samples"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)

    elif metrics == "check_full_shape":
        if X.shape == y.shape:
            message = f"✅ SAMPLE COUNT CHECK Validation is Approved. Shapes of 2 given arrays are fully consistent:\nArray 1 `shape`: {X.shape}\nArray 2 `shape`: {y.shape}"
            print_function(message)
        else:
            message = f"❌ Shapes of 2 given arrays are not consistent:\nArray 1 `shape`: {X.shape}\nArray 2 `shape`: {y.shape}"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)


def noop(*args, **kwargs):
    pass


def validate_types(
    variable: Any,
    variable_name: str,
    desired_type: type | List,
    function: Callable | None = None,
    print_function=None,
) -> None:

    if function is None:
        function = noop

    if print_function is None:
        print_function = noop

    var_type = type(variable)

    if variable is None:
        message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Approved✅"
        print_function(message)
        return
    elif isinstance(desired_type, list):
        # [int, float] cases
        if any(isinstance(variable, t) for t in desired_type):
            message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Approved✅"
            print_function(message)
        else:
            message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Dissaproved❌"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)
    elif isinstance(desired_type, type):

        if isinstance(variable, desired_type):
            # int, str cases
            message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Approved✅"
            print_function(message)
        else:

            message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Dissaproved❌"
            message = format_message(msg=message, msg_type="error")
            raise ValueError(message)
    else:
        try:
            # Simple Case
            # # Iterable cases
            if isinstance(variable, desired_type):
                message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Approved✅"
                print_function(message)
        except:

            try:
                # Rare Case
                # # Literal[1,2,3] cases
                possible_values = get_args(desired_type)
                if variable in possible_values:
                    message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Approved✅"
                    print_function(message)

                else:
                    message = f"Variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: {var_type}\nDesired type(s): {desired_type}\nValidation: Dissaproved❌"
                    message = format_message(msg=message, msg_type="error")
                    raise ValueError(message)
            except:
                message = f"Couldn't validate the data type of variable: {function.__name__}() ARGUMENT \u2192 {variable_name}\nValue Type: UNKNOWN 🛑\nDesired type(s): {desired_type}\nValidation: Dissaproved❌"
                message = format_message(msg=message, msg_type="error")
                raise ValueError(message)
    print_function("--------------")

    # print("_" * 31)


def simultaneous_shuffle(array_1: np.ndarray, array_2: np.ndarray):

    validate_shapes(array_1, array_2, metrics="check_sample_count")
    validate_types(
        variable=array_1,
        variable_name="array_1",
        desired_type=Iterable,
        function=simultaneous_shuffle,
    )
    validate_types(
        variable=array_2,
        variable_name="array_2",
        desired_type=Iterable,
        function=simultaneous_shuffle,
    )

    indices = np.arange(len(array_2))
    np.random.shuffle(indices)

    shuffled_array_1 = array_1[indices]
    shuffled_array_2 = array_2[indices]

    return shuffled_array_1, shuffled_array_2


def to_numpy(_arr: Iterable) -> np.ndarray:

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


def test_validate_types():
    # Test cases for different scenarios
    validate_types(5, "int_variable", int)
    validate_types("hello", "str_variable", str)
    validate_types([1, 2, 3], "list_variable", list)
    validate_types(5.5, "float_variable", float)
    validate_types(True, "bool_variable", bool)
    validate_types("hello", "str_variable", [int, float, str])
    validate_types(5.5, "float_variable", [int, str, list])
    validate_types(2, "int_variable", Literal[1, 2, 3])
    validate_types(4, "int_variable", Literal[1, 2, 3])


def test_validate_shapes():
    # Valid inputs with consistent shapes
    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6], [7, 8]])
    metrics = "check_sample_count"
    ignore_empty_arrays = False
    validate_shapes(array_1, array_2, metrics, ignore_empty_arrays)

    # Valid inputs with inconsistent shapes
    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6]])
    metrics = "check_sample_count"
    ignore_empty_arrays = False
    try:
        validate_shapes(array_1, array_2, metrics, ignore_empty_arrays)
    except ValueError as e:
        print(e)  # Ensure that it raises a ValueError

    # Empty input arrays (not ignoring empty arrays)
    array_1 = np.array([])
    array_2 = np.array([[5, 6]])
    metrics = "check_sample_count"
    ignore_empty_arrays = False
    try:
        validate_shapes(array_1, array_2, metrics, ignore_empty_arrays)
    except ValueError as e:
        print(e)  # Ensure that it raises a ValueError

    # Empty input arrays (ignoring empty arrays)
    array_1 = np.array([])
    array_2 = np.array([[5, 6]])
    metrics = "check_sample_count"
    ignore_empty_arrays = True
    validate_shapes(array_1, array_2, metrics, ignore_empty_arrays)

    # Valid inputs with consistent shapes (check_full_shape)
    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6], [7, 8]])
    metrics = "check_full_shape"
    ignore_empty_arrays = False
    validate_shapes(array_1, array_2, metrics, ignore_empty_arrays)

    # Valid inputs with inconsistent shapes (check_full_shape)
    array_1 = np.array([[1, 2], [3, 4]])
    array_2 = np.array([[5, 6]])
    metrics = "check_full_shape"
    ignore_empty_arrays = False
    try:
        validate_shapes(array_1, array_2, metrics, ignore_empty_arrays)
    except ValueError as e:
        print(e)  # Ensure that it raises a ValueError
