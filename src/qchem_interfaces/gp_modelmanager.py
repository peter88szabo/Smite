import os
from pathlib import Path

import torch
from torch import ScriptModule

class ModelManager:
    def __init__(self):
        self.model_directory = None
        self._models = {}

    def set_model_directory(self, model_directory: str | Path) -> None:
        if isinstance(model_directory, str):
            model_directory = Path(model_directory)
        self.model_directory = model_directory

    def load_models_from_directory(self) -> None:
        if self._models:
            return

        if not self.model_directory.is_dir():
            raise ValueError(
                f"Directory {self.model_directory} does not exist. Can't load models."
            )

        model_list = self.model_directory.iterdir()
        sorted_model_list = sorted(model for model in model_list)

        for model in sorted_model_list:
            if model.suffix in [".pt",".pth"]:
                self._models[model] = torch.jit.load(model)

    def get_models(self) -> list[ScriptModule]:
        return [*self._models.values()]


_manager = ModelManager()

# Implicit singleton implementation through the Python module system
def get_modelmanager():
    return _manager
