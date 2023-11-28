from torch.utils.data import Dataset
import os
from random import shuffle
import numpy as np
import torch


class BinaryDataset(Dataset):
    def __init__(self, goodware_directory:str=None, malware_directory:str=None, goodware_subset_filepath:str=None, malware_subset_filepath:str=None, max_len:int=4000000, sort_by_size:bool=False, padding_value:float=0.0):
        self.all_files = []
        self.max_len = max_len
        self.padding_value = padding_value

        if goodware_subset_filepath == None and malware_subset_filepath == None:
            # Add benign files
            if goodware_directory is not None:
                for roor_dir, dirs, files in os.walk(goodware_directory):
                    for file in files:
                        to_add = os.path.join(roor_dir, file)
                        self.all_files.append((to_add, 0, os.path.getsize(to_add)))

            # Add malicious files
            if malware_directory is not None:
                for roor_dir, dirs, files in os.walk(malware_directory):
                    for file in files:
                        to_add = os.path.join(roor_dir, file)
                        self.all_files.append((to_add, 1, os.path.getsize(to_add)))
        else:
            if goodware_subset_filepath is not None:
                with open(goodware_subset_filepath, "r") as goodware_subset_file:
                    lines = goodware_subset_file.readlines()
                    for line in lines[1:]:
                        to_add = os.path.join(goodware_directory, line.split(",")[0])
                        self.all_files.append((to_add, 0, os.path.getsize(to_add)))
            if malware_subset_filepath is not None:
                with open(malware_subset_filepath, "r") as malware_subset_file:
                    lines = malware_subset_file.readlines()
                    for line in lines[1:]:
                        to_add = os.path.join(malware_directory, line.split(",")[0])
                        self.all_files.append((to_add, 1, os.path.getsize(to_add)))

        if sort_by_size: #  Reorder files by size
            #sorted(self.all_files, key=lambda x: x[2])
            self.all_files.sort(key=lambda filename: filename[2])
        else: # Shuffle files
            shuffle(self.all_files)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        to_load, y, _ = self.all_files[index]

        with open(to_load, 'rb') as f:
            x = f.read(self.max_len)
            # Need to use frombuffer b/c its a byte array, otherwise np.asarray will get wonked on trying to convert to ints
            # So decode as uint8 (1 byte per value), and then convert
            if self.padding_value == 0.0:
                x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) + 1  # index 0 will be special padding index -> required for malconv and malconvgct
            else:
                x = np.frombuffer(x, dtype=np.uint8).astype(np.int16) # index 256 will be special padding index -> required for non-negative malconv

            # x = np.pad(x, self.max_len-x.shape[0], 'constant')
        x = torch.tensor(x)

        return x, torch.tensor([y])

    def pad_collate_func(self, batch):
        """
        This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to pad out files in a batch to the length of the longest item in the batch.
        """
        vecs = [x[0] for x in batch]
        labels = [x[1] for x in batch]

        x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True, padding_value=self.padding_value)
        # stack will give us (B, 1), so index [:,0] to get to just (B)
        y = torch.stack(labels)[:, 0]

        return x, y

