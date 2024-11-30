import os
import logging
import logging.config
from typing import Optional, Union
from dataclasses import dataclass
from pathlib import Path


from dotenv import load_dotenv
env_path = Path(__file__).resolve().parents[2] / ".env"

PROJECT_PATH = os.getenv("CONTAINER_PATH")

@dataclass
class CustomLogger(logging.Logger):
    config: dict[str, str]
    logger_name: Optional[str] = None
    
    def setup(self) -> logging.Logger:
        if self.logger_name is None:
            self.logger_name = __name__
    
        logger_dict = self._create_config()
        logging.config.dictConfig(logger_dict)
        return logging.getLogger(self.logger_name)

    def _create_config(self) -> dict[str, str]:
        LOGS_PATH = f"{PROJECT_PATH}/logs"
        
        logger_dict = self.config["logger"]
        
        for handler in logger_dict["handlers"]:
            if "filename" in logger_dict["handlers"][handler]:
                if not os.path.exists(LOGS_PATH):
                    os.makedirs(LOGS_PATH)
                logger_dict["handlers"][handler]["filename"] = \
                    os.path.join(LOGS_PATH, f"{self.logger_name}.log")
        
        return logger_dict