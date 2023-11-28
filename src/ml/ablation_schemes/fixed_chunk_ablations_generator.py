import torch
from typing import Tuple
import random
from src.ml.ablation_schemes.ablations_generator import AblatedEnd2EndGenerator

class FixedChunkAblatedEnd2EndGenerator(AblatedEnd2EndGenerator):
    def __init__(self, train_mode:bool=True, chunk_size:int=100000, overlapping_percentage:float=0.0, padding_value:float=0.0):
        super().__init__(train_mode)
        self.chunk_size = chunk_size
        self.overlapping_percentage = overlapping_percentage
        self.padding_value = padding_value
        self.name = "FixedChunksGenerator"

    def generate_training_chunks(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        vecs = []
        labels = []
        for x in batch:
            if x[0].shape[0] <= self.chunk_size:
                vecs.append(x[0])
            else:
                start_location = random.randint(0, max(0, x[0].shape[0] - self.chunk_size))
                end_location = start_location + self.chunk_size
                vecs.append(x[0][start_location:end_location])
            labels.append(x[1])
        x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_value)
        # stack will give us (B, 1), so index [:,0] to get to just (B)
        y = torch.stack(labels)[:, 0]

        return x, y

class SequentialFixedChunkAblatedEnd2EndGenerator(FixedChunkAblatedEnd2EndGenerator):
    def __init__(self, train_mode:bool=True, chunk_size:int=100000, overlapping_percentage:float=0.0, padding_value:float=0.0):
        FixedChunkAblatedEnd2EndGenerator.__init__(
            self,
            train_mode,
            chunk_size,
            overlapping_percentage,
            padding_value
        )
        self.name = "SequentialFixedChunkGenerator"

    def generate_testing_chunks(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        #labels = []
        if len(batch) == 1:  # Only implemented for batch sizes equals to 1
            # https://stackoverflow.com/questions/36586897/splitting-a-python-list-into-a-list-of-overlapping-chunks
            overlap_size = int(self.chunk_size * self.overlapping_percentage)
            x = [batch[0][0][i:i + self.chunk_size] for i in
                 range(0, len(batch[0][0]) - overlap_size, self.chunk_size - overlap_size)]
            x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.padding_value)
            # labels.append(batch[0][1])
            # y = torch.FloatTensor(labels)#stack(labels)[:, 0]
            return x, batch[0][1]
        else:
            raise NotImplementedError

    def generate_ablated_versions(self, x: list) -> Tuple[torch.Tensor, list]:
        overlap_size = int(self.chunk_size * self.overlapping_percentage)
        ablated_x = [x[i:i + self.chunk_size] for i in
                     range(0, len(x) - overlap_size, self.chunk_size - overlap_size)]
        locations = [[i, i + self.chunk_size] for i in
                     range(0, len(x) - overlap_size, self.chunk_size - overlap_size)]
        ablated_x = torch.nn.utils.rnn.pad_sequence(ablated_x, batch_first=True, padding_value=self.padding_value)
        return ablated_x, locations

