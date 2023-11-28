from abc import ABC, abstractmethod
import torch
from typing import Tuple

class AblatedEnd2EndGenerator(ABC):
    def __init__(self, train_mode: bool = True):
        self.train_mode = train_mode
        self.name = "AblationsGenerator"

    @abstractmethod
    def generate_training_chunks(self, batch: list):
        raise NotImplementedError

    @abstractmethod
    def generate_testing_chunks(self, batch: list):
        raise NotImplementedError

    def pad_collate_func(self, batch: list):
        """
        This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order get chunks of the file
        """
        if self.train_mode:
            x, y = self.generate_training_chunks(batch)
        else:  # Validation/test. Batch size equals to 1. For each example we might have a different number of chunks (SequentialFixedChunkGenerator)
            x, y = self.generate_testing_chunks(batch)
        return x, y

    @abstractmethod
    def generate_ablated_versions(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        raise NotImplementedError
