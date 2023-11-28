import torch
from src.ml.classifiers.MalConv import MalConv
from src.ml.end2end_builders.builder import TorchBuilder

class MalConvBuilder(TorchBuilder):
    def build_model(self) -> torch.nn.Module:
        model = MalConv(
            out_size=self.hyperparameters["out_size"],
            channels=self.hyperparameters["channels"],
            window_size=self.hyperparameters["window_size"],
            stride=self.hyperparameters["stride"],
            embd_size=self.hyperparameters["embed_size"],
            thresh=self.hyperparameters["threshold"],
            vocabulary_size=self.hyperparameters["vocabulary_size"],
            padding_idx=self.padding_idx,
        )
        return model