from abc import ABC, abstractmethod
from src.ml.datasets.binary_dataset import BinaryDataset
from src.ml.datasets.random_chunk_sampler import  RandomChunkSampler
from src.ml.ablation_schemes.fixed_chunk_ablations_generator import SequentialFixedChunkAblatedEnd2EndGenerator
from torch.utils.data import DataLoader


class DataloaderBuilder(ABC):
    def build(self):
        self.dataset = self.build_dataset()
        self.dataloader = self.build_dataloader()
        return self.dataset, self.dataloader

    @abstractmethod
    def build_dataset(self):
        pass

    @abstractmethod
    def build_dataloader(self):
        pass


class BinaryBuilder(DataloaderBuilder):
    def __init__(self, goodware_filepath: str, malware_filepath: str, goodware_subset_filepath: str = None, malware_subset_filepath: str = None, max_len: int = 16000000, sort_by_size: bool = False, padding_value: float = 0.0, num_workers: int = 1, batch_size: int = 1) -> None:
        self.goodware_filepath = goodware_filepath
        self.malware_filepath = malware_filepath
        self.goodware_subset_filepath = goodware_subset_filepath
        self.malware_subset_filepath = malware_subset_filepath
        self.max_len = max_len
        self.sort_by_size = sort_by_size
        self.padding_value = padding_value
        self.num_workers = num_workers
        self.batch_size = batch_size

    def build_dataset(self):
        dataset = BinaryDataset(
            self.goodware_filepath,
            self.malware_filepath,
            goodware_subset_filepath=self.goodware_subset_filepath,
            malware_subset_filepath=self.malware_subset_filepath,
            max_len=self.max_len,
            sort_by_size=self.sort_by_size,
            padding_value=self.padding_value
        )
        return dataset

    def build_dataloader(self):
        if self.sort_by_size is False:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.dataset.pad_collate_func
            )
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                collate_fn=self.dataset.pad_collate_func,
                sampler=RandomChunkSampler(self.dataset, self.batch_size))
        return dataloader


class FixedChunkAblationsBinaryBuilder(BinaryBuilder):
    def __init__(self, goodware_filepath: str, malware_filepath: str, goodware_subset_filepath: str = None, malware_subset_filepath: str = None, max_len: int = 16000000, sort_by_size: bool = False, padding_value: float = 0.0, num_workers: int = 1, batch_size: int = 1, chunk_size: int = 500, train_mode: bool = True) -> None:
        BinaryBuilder.__init__(
            self,
            goodware_filepath,
            malware_filepath,
            goodware_subset_filepath,
            malware_subset_filepath,
            max_len,
            sort_by_size,
            padding_value,
            num_workers,
            batch_size
        )
        self.chunk_size = chunk_size
        self.train_mode = train_mode

    def build_dataloader(self):
        self.ablated_generator = self.initialize_fixed_chunk_ablations_generator()
        if self.train_mode is True:
            if self.sort_by_size is False:
                dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.ablated_generator.pad_collate_func
                )
            else:
                dataloader = DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.ablated_generator.pad_collate_func,
                    sampler=RandomChunkSampler(self.dataset, self.batch_size))
        else:
            dataloader = DataLoader(
                self.dataset,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=self.ablated_generator.pad_collate_func,
            )
        return dataloader

    @abstractmethod
    def initialize_fixed_chunk_ablations_generator(self):
        pass




class SequentialFixedChunkAblationsBinaryBuilder(FixedChunkAblationsBinaryBuilder):
    def __init__(self, goodware_filepath: str, malware_filepath: str, goodware_subset_filepath: str = None,
                 malware_subset_filepath: str = None, max_len: int = 16000000, sort_by_size: bool = False,
                 padding_value: float = 0.0, num_workers: int = 1, batch_size: int = 1, chunk_size: int = 500,
                 train_mode: bool = True) -> None:
        FixedChunkAblationsBinaryBuilder.__init__(
            self,
            goodware_filepath,
            malware_filepath,
            goodware_subset_filepath,
            malware_subset_filepath,
            max_len,
            sort_by_size,
            padding_value,
            num_workers,
            batch_size,
            chunk_size,
            train_mode
        )


    def initialize_fixed_chunk_ablations_generator(self):
        ablated_generator = SequentialFixedChunkAblatedEnd2EndGenerator(
            train_mode=self.train_mode,
            chunk_size=self.chunk_size,
            padding_value=self.padding_value
        )
        return ablated_generator




