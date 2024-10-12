import os
import logging
import logging.config
from typing import Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from pathlib import Path

from dotenv import load_dotenv
env_path = Path(__file__).resolve().parents[1] / '.devcontainer' / '.env'
load_dotenv(dotenv_path=env_path)

PROJECT_PATH = os.getenv("CONTAINER_PATH")

@dataclass
class CustomLogger(logging.Logger):
    """
    Creates and sets up a logger instance.
    """
    config_dict: dict
    logger_name: Optional[str] = None

    def setup_logger(self) -> logging.Logger:
        """Function to setup a logger."""
        if self.logger_name is None:
            self.logger_name = __name__

        logger_dict = self._create_logger_config()
        logging.config.dictConfig(logger_dict)
        return logging.getLogger(self.logger_name)

    def _create_logger_config(self) -> dict:
        """Loads a configuration file for the logger."""
        LOGS_PATH = f"{PROJECT_PATH}/logs"

        logger_dict = self.config_dict["logger"]

        for handler in logger_dict["handlers"]:
            if "filename" in logger_dict["handlers"][handler]:

                if not os.path.exists(LOGS_PATH):
                    os.makedirs(LOGS_PATH)

                logger_dict["handlers"][handler]["filename"] = \
                    logger_dict["handlers"][handler]["filename"] = \
                        os.path.join(LOGS_PATH, f"{self.logger_name}.log")
        return logger_dict


def log_function(
    config_dict: dict,
    logger: Optional[logging.Logger] = None
):
    """Decorator to log the function name and the arguments passed to it."""
    if logger is None:
        logger = CustomLogger(config_dict=config_dict).setup_logger()

    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with arguments {args} and {kwargs}")
            return func(*args, **kwargs)
        return wrapper
    return decorator