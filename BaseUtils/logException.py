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

"""
import os 
import sys
import time
import colorama
import traceback


class LogException:
    Header = """Logger V1.0.0 
This is the log file for recording exceptions and errors."""


    def __init__(self,
                 base_path: str ):
        """
        Initializes the LogException class. Creates a log file if it doesn't exist.
        Args:
            base_path (str): The base directory where the log file will be created.
        """
        if not os.path.exists(base_path):
            raise FileNotFoundError(colorama.Fore.RED + f"The specified base path does not exist: {base_path}" + colorama.Style.RESET_ALL)

        log_file_path = os.path.join(base_path, f"log_{time.strftime('%Y-%m-%d %H:%M:%S')}.log")
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w") as log_file:
                log_file.write(f"{self.Header}\n")
        self.log_file_path = log_file_path

    def _message(self,
                 type: str,
                 message: str) -> None:
        """
        Logs a message to the log file.
        """
        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
            log_file.write(f"{type.upper()}: {message}\n")
            log_file.write("\n" + "="*60 + "\n")

    def warning_message(self,
                        message: str) -> None:
        self._message("WARNING", message)


    def success_message(self,
                        message: str) -> None:
        self._message("SUCCESS", message)   

    def log_exception(self, 
                      e: Exception,
                      custom_message: str = "",
                      Verbose: bool = False
                      ) -> None:
        """
        Logs an exception with an optional custom message.
        
        Args:
            e (Exception): The exception to log.
            custom_message (str): An optional custom message to include in the log.
        """


        exc_type, exc_value, exc_traceback  = sys.exc_info()
        tb_lines                            = traceback.format_exception(exc_type, exc_value, exc_traceback)
        tb_text                             = ''.join(tb_lines)

        if Verbose:
            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
            if custom_message:
                print(f"Custom Message: {custom_message}")
            
            print(f"Exception Type: {exc_type.__name__ if exc_type is not None else 'Unknown'}")
            print(f"Exception Message: {str(e)}")
            print("Traceback:")
            print(tb_text)
            print(colorama.Fore.RED + "An error occurred. Check log file for details." + colorama.Style.RESET_ALL)


        with open(self.log_file_path, "a") as log_file:
            log_file.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
            if custom_message:
                log_file.write(f"Custom Message: {custom_message}\n")
            log_file.write(f"Exception Type: {exc_type.__name__ if exc_type is not None else 'Unknown'}\n")
            log_file.write(f"Exception Message: {str(e)}\n")
            log_file.write("Traceback:\n")
            log_file.write(tb_text)
            log_file.write("\n" + "="*60 + "\n")


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