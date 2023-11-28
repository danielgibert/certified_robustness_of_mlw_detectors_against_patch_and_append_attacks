from __future__ import annotations
from abc import ABC, abstractmethod
import json
import torch
from src.utils import get_pretrained_embeddings


class TorchBuilder(ABC):
    """
    The Builder interface specifies methods to build the MalConv-based models
    """
    def build(self, hyperparameters_filepath: str, model_checkpoint: str = None, pretrained_emb: str = None, device: torch.device = None, padding_idx: int = 0) -> (torch.nn.Module, dict):
        self.device = device
        self.hyperparameters = self.load_hyperparameters(hyperparameters_filepath)
        self.padding_idx = padding_idx
        self.model = self.build_model()
        self.restore_checkpoint(model_checkpoint)
        self.freeze_pretrained_embedding(pretrained_emb)
        self.model.to(self.device)
        return self.model, self.hyperparameters

    def load_hyperparameters(self, hyperparameters_filepath: str) -> dict:
        with open(hyperparameters_filepath, "r") as hyperparameters_file:
            hyperparameters = json.load(hyperparameters_file)
        return hyperparameters

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        pass

    def restore_checkpoint(self, model_checkpoint: str) -> None:
        if model_checkpoint is not None:
            model_checkpoint = torch.load(model_checkpoint, map_location=self.device)
            self.model.load_state_dict(model_checkpoint['model_state_dict'], strict=False)

    def freeze_pretrained_embedding(self, pretrained_emb: str = None) -> None:
        if pretrained_emb is not None:
            embd_weights_dict = get_pretrained_embeddings(pretrained_emb)
            self.model.update_embedding_weights(embd_weights_dict, padding_idx=self.padding_idx)



