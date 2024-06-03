#!/bin/bash

# WARNING: USE WITH CAUTION this script is meant to clean any folder of models's subfolder for all the models in that folder.
# For example if parent folder is exp/cifar10, and the target_subsubfolder is "logits", it will remove for EACH model the computed logits.

parent_folder="exp/cifar10_noisy_members"
target_subsubfolder="logits"

# Find and remove target subsubfolder
find "$parent_folder" -type d -name "$target_subsubfolder" -exec rm -rf {} \;