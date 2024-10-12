import os
import toml
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

MAIN_PATH = Path(__file__).resolve().parents[1]
CONFIGURATION_PATH = Path(MAIN_PATH, "config")


class FileHandler(ABC):
    @abstractmethod
    def load(self) -> dict:
        pass

    @abstractmethod
    def save(self, config_dict: dict) -> None:
        pass

@dataclass
class TomlHandler(FileHandler):
    filename: str

    def __post_init__(self):
        if not os.path.exists(CONFIGURATION_PATH):
            os.makedirs(CONFIGURATION_PATH)

    def load(self) -> dict:
        try:
            with open(f"{os.path.join(CONFIGURATION_PATH, self.filename)}", "r") as f:
                return toml.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {self.filename} not found in {CONFIGURATION_PATH}.") from e

    def save(self, config_dict: dict) -> None:
        with open(f"{os.path.join(CONFIGURATION_PATH, self.filename)}", "w") as f:
            toml.dump(config_dict, f)