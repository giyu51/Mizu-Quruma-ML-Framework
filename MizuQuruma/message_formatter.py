from typing import Literal
import colorama
from colorama import Fore, Style

colorama.init()


def format_message(
    msg: str, msg_type: Literal["error", "warning", "info"] = "info"
) -> None:
    """
    Display a formatted message with the specified message type.

    Args:
        msg (str): The message to be displayed.
        msg_type (Literal["error", "warning", "info"], optional):
            The type of message. Defaults to "info".
            Valid options are:
                - "error": An error message.
                - "warning": A warning message.
                - "info": A regular informational message.

    Raises:
        ValueError: If an unsupported `msg_type` is provided.

    Returns:
        None
    """
    upper_line = "\u2500" * 20
    small_line = "\u2500" * 1
    lower_line = "\u2500" * 10

    if msg_type == "info":
        formatted_message = f"{Fore.GREEN}\n\n{small_line} INFO: {upper_line}\n\n{msg}\n\n{lower_line}\n{Style.RESET_ALL}"
        return formatted_message
    elif msg_type == "error":
        formatted_message = f"{Fore.RED}\n\n{small_line} ERROR: {upper_line}\n\n{msg}\n\n{lower_line}\n{Style.RESET_ALL}"
        return formatted_message
    elif msg_type == "warning":
        formatted_message = f"{Fore.YELLOW}\n\n{small_line} WARNING: {upper_line}\n\n{msg}\n\n{lower_line}\n{Style.RESET_ALL}"
        return formatted_message
    else:
        raise ValueError(
            f"Invalid msg_type: {msg_type}. Expected one of ['error', 'warning', 'info']"
        )


if __name__ == "__main__":
    message = "Testing all message types:\n1.Info\n2.Error\n3.Warning"
    format_message(message, msg_type="info")
    format_message(message, msg_type="error")
    format_message(message, msg_type="warning")
    # Uncomment the following line to test handling of unexpected msg_type
    # display_message(message, msg_type="info")
