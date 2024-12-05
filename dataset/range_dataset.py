from itertools import chain

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from range_samplers import *


class RangeSampler:
    def __init__(self, range_fn: str, sample_size: int, config: dict):
        self.range_fn = range_fn
        self.sample_size = sample_size
        self.config = config

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
            return range_center
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
            mask_model = self.config["ramia"].get("mask_model", None)
            if mask_model is None:
                raise ValueError(
                    "Word replace range sampler requires a mask_model parameter in the config."
                )
            mask_tokenizer = self.config["ramia"].get("mask_tokenizer", mask_model)
            num_masks = self.config["ramia"].get("num_masks", None)
            if num_masks is None:
                raise ValueError(
                    "Word replace range sampler requires a num_masks parameter in the config."
                )
            device = self.config["ramia"].get("device", "cuda")
            return sample_word_replace(
                range_center,
                mask_model,
                mask_tokenizer,
                num_masks,
                self.sample_size,
                device,
            )
        # elif self.range_fn == "ownership":
        #     ownership_dict_path = self.config["ramia"].get("ownership_dict_path", None)
        #     if ownership_dict_path is None:
        #         raise ValueError(
        #             "Ownership range sampler requires an ownership_dict_path parameter in the config."
        #         )
        #     return sample_ownership(range_center, ownership_dict_path, self.sample_size)
        # elif self.range_fn == "missing_features":
        #     missing_features = self.config["ramia"].get("missing_features", None)
        #     if missing_features is None:
        #         raise ValueError(
        #             "Missing features range sampler requires a missing_features parameter in the config."
        #         )
        #     return sample_missing_features(range_center, missing_features, self.sample_size)
        else:
            raise ValueError(f"Range function {self.range_fn} is not supported.")


class RangeDataset(Dataset):
    def __init__(self, dataset: Dataset, sampler: RangeSampler, config: dict):
        self.dataset = dataset
        self.sampler = sampler
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.sampler.range_fn == "word_replace" and hasattr(
            self.dataset, "get_text"
        ):
            # Determining if it is a text dataset
            text = self.dataset.get_text(idx)
            if self.sampler.range_fn != "word_replace":
                raise ValueError("Range sampler is not compatible with text data.")
            range_text = self.sampler.sample(text)
            tokenizer = AutoTokenizer.from_pretrained(self.config["data"]["tokenizer"])
            range_data = tokenizer(
                list(chain.from_iterable(range_text)),
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            data = range_data.input_ids[idx][:-1]
            target = range_data.input_ids[idx][1:]
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
