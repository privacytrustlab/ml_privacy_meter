import torch
from torch.utils.data import Dataset

from range_samplers import *


class RangeSampler:
    def __init__(self, range_fn: str, sample_size: int, config: dict):
        self.range_fn = range_fn
        self.sample_size = sample_size
        self.config = config

    def sample(self, range_center):
        if self.sample_size == 1:
            print("Sample size is 1, returning range center.")
            return range_center
        elif self.sample_size < 1:
            raise ValueError("Sample size must be greater than 0.")

        if self.range_fn == "l2":
            radius = self.config["ramia"].get("radius", None)
            if radius is None:
                raise ValueError("L2 range sampler requires a radius parameter in the config.")
            return sample_l2(range_center, radius, self.sample_size)
        elif self.range_fn == "geometric":
            transformations_list = self.config["ramia"].get("transformations", None)
            if transformations_list is None:
                raise ValueError(
                    "Geometric range sampler requires a transformations parameter in the config."
                )
            if len(transformations_list) == 0:
                raise ValueError("Transformations list cannot be empty.")
            elif len(transformations_list) != self.sample_size - 1 and len(transformations_list) != self.sample_size:
                raise ValueError("Transformations list must have length sample_size - 1 or sample_size.")
            return sample_geometric(range_center, transformations_list, self.sample_size)
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
            return sample_word_replace(range_center, mask_model, mask_tokenizer, num_masks, self.sample_size, device)
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
    def __init__(self, dataset: Dataset, sampler: RangeSampler):
        self.dataset = dataset
        self.sampler = sampler

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # Determining if it is a text dataset
            text = self.dataset.get_text(idx)
            if self.sampler.range_fn != "word_replace":
                raise ValueError("Range sampler is not compatible with text data.")
            range_text = self.sampler.sample(text)
        except:
            range_data = self.sampler.sample(self.dataset[idx][0])
            range_labels = torch.tensor(self.dataset[idx][1], dtype=torch.long).repeat(range_data.shape[0], 1)
            return torch.tensor(range_data, dtype=torch.float32), range_labels
