import pickle
from typing import Any


def save_instance(instance: Any, filename: str = "model.pkl") -> None:
    """
    Save an instance to a file using pickle serialization.

    Args🛠️:
        - `instance` (Any): The instance to be saved.
        - `filename` (str): The filename to save the instance to. Default is 'model.pkl'.

    ---

    Example 1 💡:
        >>> # Create an instance of a model (e.g., KNN)
        >>> model = KNN()
        >>> # Fit the model to training data
        >>> model.fit(X_train, y_train)
        >>> # Save the fitted model to a file
        >>> save_instance(model, "knn_model_1.pkl")

    Example 2 💡:
        >>> # Create an instance of a LabelEncoder (e.g., StringToIntEncoder)
        >>> enc = StringToIntEncoder()
        >>> # Fit the encoder to vocabulary
        >>> enc.fit(vocabulary)
        >>> # Save the fitted encoder to a file
        >>> save_instance(enc, "encoder_1.pkl")

    ---

    Raises ⛔:
        - `TypeError`: If the instance cannot be serialized using pickle.
        - `FileNotFoundError`: If the specified file or directory does not exist.

    ---

    Note⚠️:
        The instance will be saved using pickle serialization, which may not be secure if
        loading untrusted data. Exercise caution when loading instances from untrusted sources.
    """
    with open(filename, "wb") as file:
        pickle.dump(instance, file)


def load_instance(filename: str = "model.pkl") -> Any:
    """
    Load an instance from a file using pickle deserialization.

    Args🛠️:
        - `filename` (str): The filename to load the instance from. Default is 'model.pkl'.

    ---

    Returns📤:
        - `Any`: The loaded instance.

    ---

    Example 1 💡:
        >>> # Load a saved model instance (e.g., KNN)
        >>> loaded_model = load_instance("knn_model_1.pkl")

    Example 2 💡:
        >>> # Load a saved LabelEncoder instance (e.g., StringToIntEncoder)
        >>> loaded_enc = load_instance("encoder_1.pkl")

    ---

    Raises ⛔:
        - `EOFError`: If the end of the file is reached unexpectedly.
        - `TypeError`: If the data in the file cannot be deserialized using pickle.
        - `FileNotFoundError`: If the specified file does not exist.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)
