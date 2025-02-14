import os
from pathlib import Path

import torch
from torch import ScriptModule

model_directory = "./models"


class ModelManager:
    def __init__(self):
        self._models = {}

    def load_models_from_directory(self):
        if self._models:
            return

        global model_directory
        if not isinstance(model_directory, Path):
            model_directory = Path(model_directory)

        if not model_directory.is_dir():
            raise ValueError(
                f"Directory {model_directory} does not exist. Can't load models."
            )

        model_list = os.listdir(model_directory)
        model_list = sorted([Path(model) for model in model_list])

        for model in model_list:
            if model.suffix == ".pt" or model.suffix == ".pth":
                self._models[model] = torch.jit.load(model_directory / model)

    def get_models(self) -> list[ScriptModule]:
        return [*self._models.values()]


_manager = ModelManager()
_manager.load_models_from_directory()


# Implicit singleton implementation through the Python module system
def get_modelmanager():
    return _manager
