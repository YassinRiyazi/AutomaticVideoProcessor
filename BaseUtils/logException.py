"""
    Version             : 1.0.0

    Author              : Yassin Riyazi
    Date                : 03.10.2025
    Project             : Automatic Video Processor (AVP)
    File                : BaseUtils/logException.py
    License             : GNU GENERAL PUBLIC LICENSE Version 3
    Level access in API : level 0 utility
    Copy Right          : Max Planck Institute for Polymer Research 2025©

    Description: 
        This module provides a utility class for logging exceptions with detailed traceback information.

    Note:
        Will not handle 
            1. FileNotFoundError

    Changelog:
        V1.0.1 29.10.2025 : Minor bug fix, problem of making name file in windows.
        V1.0.0 03.10.2025 : Initial version.

"""
import os 
import sys
import time
import colorama
import traceback


class LogException:
    """
    A lightweight exception and message logger.

    This class creates timestamped log files for storing exceptions, warnings, and success messages.

    Example:
        logger = LogException(base_path="D:/Logs")
        try:
            raise ValueError("Example error")
        except Exception as e:
            logger.log_exception(e, custom_message="Something went wrong!", Verbose=True)
    """

    HEADER = (
        "Logger V1.0.0\n"
        "This is the log file for recording exceptions and errors.\n"
        + "=" * 60 + "\n"
    )

    def __init__(self, base_path: str):
        """
        Initializes the LogException class and creates a log file if it doesn't exist.

        Args:
            base_path (str): The base directory where the log file will be created.

        Raises:
            FileNotFoundError: If the base path does not exist.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                colorama.Fore.RED
                + f"The specified base path does not exist: {base_path}"
                + colorama.Style.RESET_ALL
            )

        # Safe filename for Windows (replace ':' and spaces)
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file_path = os.path.join(base_path, f"log_{timestamp}.log")

        # Create a new log file with header
        with open(self.log_file_path, "w", encoding="utf-8") as log_file:
            log_file.write(self.HEADER)

    def _write_message(self, level: str, message: str) -> None:
        """Writes a formatted message to the log file."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}]\n{level.upper()}: {message}\n")
            log_file.write("=" * 60 + "\n")

    def warning_message(self, message: str) -> None:
        """Logs a warning message."""
        self._write_message("WARNING", message)

    def success_message(self, message: str) -> None:
        """Logs a success message."""
        self._write_message("SUCCESS", message)

    def log_exception(
        self, e: Exception, custom_message: str = "", Verbose: bool = False
    ) -> None:
        """
        Logs an exception with an optional custom message.

        Args:
            e (Exception): The exception to log.
            custom_message (str, optional): A message to provide additional context.
            Verbose (bool, optional): If True, also prints the exception details to the console.
        """
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        log_entry = [f"[{timestamp}]"]
        if custom_message:
            log_entry.append(f"Custom Message: {custom_message}")
        log_entry.append(f"Exception Type: {exc_type.__name__ if exc_type else 'Unknown'}")
        log_entry.append(f"Exception Message: {str(e)}")
        log_entry.append("Traceback:")
        log_entry.append(tb_text)
        log_entry.append("=" * 60)

        with open(self.log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n".join(log_entry) + "\n")

        if Verbose:
            print("\n".join(log_entry))
            print(colorama.Fore.RED + "An error occurred. Check log file for details." + colorama.Style.RESET_ALL)


if __name__ == "__main__":
    # Example usage
    logger = LogException(base_path=".")
    try:
        test = 1 / 0
        del test

    except ZeroDivisionError:
        logger.warning_message("This is a division by zero error")

    except Exception as e:
        if e == ZeroDivisionError("division by zero"):
            logger.warning_message("This is a warning message.")
        elif e == NameError("name 'test' is not defined"):
            logger.success_message("This is a success message.")
        else:
            logger.log_exception(e, custom_message="Unknown/New error occurred", Verbose=False)
    logger.success_message("This is a success message.")