from typing import Tuple, Union

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

"""
This dataset class will be mainly used in inference because it follows the same format as image and tabular datasets.
In training of LLMs, standard Huggingface pipelines are preferred, where HFDataset is used.
"""


class TextDataset(Dataset):
    def __init__(
        self, hf_dataset: HFDataset, target_column: str, text_column: str = "text"
    ):
        """
        Initialize the TextDataset.

        Args:
            hf_dataset (HFDataset): Hugging Face dataset object.
            target_column (str): Column name to use as target.
            text_column (str): Column name to use as text feature (default is 'text').
        """
        self.hf_dataset = hf_dataset
        self.texts = self.hf_dataset[text_column]
        self.input_ids = self.hf_dataset[target_column]

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.hf_dataset)

    def __getitem__(
        self, idx: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the input and target tensors for a given index.

        Args:
            idx (Union[int, torch.Tensor]): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
        """
        # Convert idx from tensor to list if necessary (e.g., for pytorch random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        # Get the input ids and shifted ids as targets at the given index
        inputs = self.input_ids[idx][:-1]
        target = self.input_ids[idx][1:]

        return torch.tensor(inputs), torch.tensor(target)

    def get_text(self, idx: int) -> str:
        """
        Get the text at a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            str: Text at the given index.
        """
        return self.texts[idx]
