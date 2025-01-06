import pdb
from itertools import chain

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

from range_samplers import *


class RangeSampler:
    def __init__(self, range_fn: str, sample_size: int, config: dict):
        self.range_fn = range_fn
        self.sample_size = sample_size
        self.config = config

        if self.range_fn == "word_replace":
            # Initialize tokenizer and model only once in the main thread
            self.mask_model = self.config["ramia"].get("mask_model", None)
            self.mask_tokenizer = self.config["ramia"].get(
                "mask_tokenizer", self.mask_model
            )
            self.num_masks = self.config["ramia"].get("num_masks", None)
            self.device = self.config["ramia"].get("device", "cuda")

            # Load model and tokenizer outside of sampling function to avoid reloading during each sample
            if self.mask_model is None:
                raise ValueError(
                    "Word replace range sampler requires a mask_model parameter in the config."
                )
            if self.num_masks is None:
                raise ValueError(
                    "Word replace range sampler requires a num_masks parameter in the config."
                )

            # Load the masked language model only once
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(self.mask_model).to(
                self.device
            )
            self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.mask_tokenizer)

    def sample(self, range_centers):
        # samples = []
        # for range_center in range_centers:
        #     print(range_center.shape)
        #     samples.append(self._sample(range_center))
        # return samples
        return self._sample(range_centers)

    def _sample(self, range_center):
        if self.sample_size == 1:
            print("Sample size is 1, returning range center.")
            return [range_center]
        elif self.sample_size < 1:
            raise ValueError("Sample size must be greater than 0.")

        if self.range_fn == "l2":
            radius = self.config["ramia"].get("radius", None)
            if radius is None:
                raise ValueError(
                    "L2 range sampler requires a radius parameter in the config."
                )
            return sample_l2(range_center, radius, self.sample_size)
        elif self.range_fn == "geometric":
            transformations_list = self.config["ramia"].get("transformations", None)
            if transformations_list is None:
                raise ValueError(
                    "Geometric range sampler requires a transformations parameter in the config."
                )
            if len(transformations_list) == 0:
                raise ValueError("Transformations list cannot be empty.")
            elif (
                len(transformations_list) != self.sample_size - 1
                and len(transformations_list) != self.sample_size
            ):
                raise ValueError(
                    f"Transformations list must have length sample_size - 1 or sample_size. Current transformations list "
                    f"length: {len(transformations_list)}, sample size {self.sample_size}"
                )
            return sample_geometric(
                range_center, transformations_list, self.sample_size
            )
        elif self.range_fn == "word_replace":
            return sample_word_replace(
                range_center,
                self.mlm_model,
                self.mlm_tokenizer,
                self.num_masks,
                self.sample_size,
                self.device,
            )
        else:
            raise ValueError(f"Range function {self.range_fn} is not implemented.")


class RangeDataset(Dataset):
    def __init__(self, dataset: Dataset, sampler: RangeSampler, config: dict):
        self.dataset = dataset
        self.sampler = sampler
        self.config = config
        self.tokenizer = (
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer"])
            if self.config["data"].get("tokenizer", None) is not None
            else None
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sampler.range_fn == "word_replace":
            if hasattr(self.dataset, "get_text"):
                # Determining if it is a text dataset
                text = self.dataset.get_text(idx)
            else:
                raise ValueError(
                    "The underlying dataset does not have a get_text method. Please check the implementation."
                )
            range_text = self.sampler.sample(text)
            # tokenizer = AutoTokenizer.from_pretrained(self.config["data"]["tokenizer"])
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            range_data = self.tokenizer(
                # list(chain.from_iterable(range_text)),
                range_text,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            data = range_data.input_ids[:, :-1]
            target = range_data.input_ids[:, 1:]
            return data, target
        else:
            range_data = self.sampler.sample(self.dataset[idx][0])
            # print("Length of range_data is: ", len(range_data))
            # print("The length of the first range data is: ", range_data[0].shape)
            if len(range_data) == 1:
                range_data = range_data[0]
            else:
                range_data = torch.stack(range_data)
            # print("Shape of the stacked range data is: ", range_data.shape)
            if type(self.dataset[idx][1]) == int:
                range_labels = torch.tensor(
                    self.dataset[idx][1], dtype=torch.long
                ).repeat(self.config["ramia"]["sample_size"])
            else:
                range_labels = torch.tensor(self.dataset[idx][1]).repeat_interleave(
                    self.config["ramia"]["sample_size"]
                )
            return range_data, range_labels
