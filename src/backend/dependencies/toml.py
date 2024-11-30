import os
import toml
from dataclasses import dataclass
from pathlib import Path

BACKEND_PATH = Path(__file__).resolve().parents[1]
CONFIGURATION_PATH = BACKEND_PATH / "config"

@dataclass
class TomlHandler:
    filename: str
    
    def __post_init__(self):
        if not os.path.exists(CONFIGURATION_PATH):
            os.makedirs(CONFIGURATION_PATH)
            
    def load(self) -> dict[str, str]:
        try:
            with open(f"{os.path.join(CONFIGURATION_PATH, self.filename)}", "r") as f:
                return toml.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File {self.filename} not found in {CONFIGURATION_PATH}.") from e

    def save(self, data: dict[str, str]) -> None:
        with open(f"{os.path.join(CONFIGURATION_PATH, self.filename)}", "w") as f:
            toml.dump(data, f)