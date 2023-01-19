import collections
import copy
import logging
import pickle
import time
from ast import List
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from dataset import get_dataset_subset
from models import get_model
from privacy_meter import audit_report
from privacy_meter.audit import MetricEnum
from privacy_meter.audit_report import ROCCurveReport, SignalHistogramReport
from privacy_meter.constants import InferenceGame
from privacy_meter.dataset import Dataset
from privacy_meter.information_source import InformationSource
from privacy_meter.model import PytorchModelTensor
from train import inference, train
from util import get_split, load_models_by_conditions, load_models_by_model_idx


def load_existing_target_model(dataset_size, model_metadata_dict, configs):
    """Check if there are trained models that matches the training configuration.

    Args:
        dataset_size (int): Size of the whole training dataset.
        model_metadata_dict (dict): Model metedata dict.
        configs (dict): Training target models configuration.

    Raises:
        ValueError: Check if the key for the configuration takes value from data_idx and model_idx

    Returns:
        matched_idx: List of target model index which matches the conditions
    """
    assert isinstance(model_metadata_dict, dict)
    assert "model_metadata" in model_metadata_dict

    matched_idx_list = []
    # Specify the conditions.
    num_target_models = configs["train"].get("num_target_model", 1)
    if configs["train"]["key"] == "none":
        conditions = {
            "optimizer": configs["train"]["optimizer"],
            "batch_size": configs["train"]["batch_size"],
            "epochs": configs["train"]["epochs"],
            "learning_rate": configs["train"]["learning_rate"],
            "weight_decay": configs["train"]["weight_decay"],
            "num_train": int(dataset_size * configs["data"]["f_train"]),
        }
        matched_idx_list = load_models_by_conditions(
            model_metadata_dict, conditions, num_target_models
        )
    elif configs["train"]["key"] == "model_idx":
        matched_idx_list = load_models_by_model_idx(
            model_metadata_dict, [configs["train"]["idx"]]
        )
    else:
        raise ValueError("key can only be none or model_idx for loading target models")
    return matched_idx_list


def load_existing_reference_models(
    dataset_size, model_metadata_dict, configs, matched_target_idx
):
    """Check if there are trained models that matches the configuration.

    Args:
        dataset_size (int): Size of the whole training dataset.
        model_metadata_dict (dict): Model metedata dict.
        configs (dict): Training target models configuration.
        matched_target_idx (List): List of existing target model index.

    Raises:
        ValueError: If the key for the configuration takes value from none and model_idx

    Returns:
        reference_matched_idx_list: List of reference model index which matches the conditions
    """
    assert isinstance(model_metadata_dict, dict)
    assert "model_metadata" in model_metadata_dict
    assert isinstance(matched_target_idx, list)
    if configs["audit"]["algorithm"] != "reference":
        return []
    # Specify the conditions.
    matched_idx_list = []
    number_models = configs["audit"].get("num_reference_model", 10)
    if configs["audit"]["key"] == "none":
        conditions = {
            "optimizer": configs["train"]["optimizer"],
            "batch_size": configs["train"]["batch_size"],
            "epochs": configs["train"]["epochs"],
            "learning_rate": configs["train"]["learning_rate"],
            "weight_decay": configs["train"]["weight_decay"],
            "num_train": int(dataset_size * configs["data"]["f_train"]),
        }
        for target_idx in matched_target_idx:
            reference_matched_idx = load_models_by_conditions(
                model_metadata_dict, conditions, number_models, [target_idx]
            )
            matched_idx_list.append(reference_matched_idx)
    else:
        raise ValueError("key can only be none or model_idx for loading target models")
    return matched_idx_list


def load_existing_models(
    model_metadata_dict: dict, matched_idx: List(int), model_name: str
):
    model_list = []
    if len(matched_idx) > 0:
        for metadata_idx in matched_idx:
            metadata = model_metadata_dict["model_metadata"][metadata_idx]
            model = get_model(model_name)
            with open(f"{metadata['model_path']}", "rb") as file:
                model_weight = pickle.load(file)
            model.load_state_dict(model_weight)
            model_list.append(model)
        print(f"Load existing {len(model_list)} target models")
    else:
        return []


def load_dataset_for_existing_models(
    dataset_size: int, model_metadata_dict: dict, matched_idx: List(int), configs: dict
) -> List(dict):
    """Load the dataset index list for the existing models.

    Args:
        dataset_size (int): Dataset size.
        model_metadata_dict (dict): Model metadata dict.
        matched_idx (List(int)): List of matched model index.
        configs (dict): Training configuration.
    Returns:
        _type_: _description_
    """
    assert isinstance(matched_idx, list)
    all_index = np.arange(dataset_size)
    test_size = int(configs["f_test"] * dataset_size)
    audit_size = int(configs["f_audit"] * dataset_size)
    index_list = []
    for metadata_idx in matched_idx:
        metadata = model_metadata_dict["model_metadata"][metadata_idx]
        # Check if the existing target data has the test split.
        if "test_split" in metadata:
            index_list.append(
                {
                    "train": metadata["train_split"],
                    "test": metadata["test_split"],
                    "audit": metadata["audit_split"],
                }
            )
        else:
            all_index = np.range(dataset_size)
            test_index = get_split(
                all_index, metadata["train_split"], test_size, "no_overlapping"
            )
            used_index = np.concatenate([metadata["train_split"], test_index])
            audit_index = get_split(
                all_index,
                used_index,
                size=audit_size,
                split_method=configs["split_method"],
            )
            index_list.append(
                {
                    "train": metadata["train_split"],
                    "test": test_index,
                    "audit": audit_index,
                }
            )
    return index_list


def prepare_datasets(dataset_size: int, num_datasets: int, configs: dict):
    """Prepare the dataset for training the target models when the training data are sampled uniformly from the distribution (pool of all possible data).

    Args:
        dataset_size (int): Size of the whole dataset
        num_datasets (int): Number of datasets we should generate
        configs (dict): Data split configuration

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        dataset_splits: Data split information which saves the information of training points index and test points index for all target models.
    """

    # The index_list will save all the information about the train, test and auit for each target model.
    index_list = []
    all_index = np.arange(dataset_size)
    train_size = int(configs["f_train"] * dataset_size)
    test_size = int(configs["f_test"] * dataset_size)
    audit_size = int(configs["f_audit"] * dataset_size)
    for _ in range(num_datasets):
        selected_index = np.random.choice(
            all_index, train_size + test_size, replace=False
        )
        train_index, test_index = train_test_split(selected_index, test_size=test_size)
        audit_index = get_split(
            all_index,
            selected_index,
            size=audit_size,
            split_method=configs["split_method"],
        )
        index_list.append(
            {"train": train_index, "test": test_index, "audit": audit_index}
        )

    dataset_splits = {"split": index_list, "split_method": configs["split_method"]}
    return dataset_splits


def prepare_datasets_for_sample_privacy_risk(
    dataset_size,
    num_total,
    num_models,
    data_idx,
    configs,
    data_type,
    model_metadata_dict,
    matched_in_idx=None,
):
    """Prepare the datasets for auditing the priavcy risk for a data point. We prepare the dataset with or without the target point for training a set of models with or without the target point.

    Args:
        dataset_size (int): Size of the whole dataset
        num_total (int): Number of all target models
        num_models (int): Number of additional target models
        data_idx (int): Data index of the target point
        configs (dict): Data split configuration
        data_type (str): Indicate whether we want to include the target point or exclude the data point (takes value from 'include' and 'exclude' )
        model_metadata_dict (dict): Metadata for existing models.
        matched_in_idx (list, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    associated_models = []
    all_index = np.arange(dataset_size)
    all_index_exclude_z = np.array([i for i in all_index if i != data_idx])
    index_list = []

    # Indicate how to sample the rest of the dataset.
    if configs["split_method"] == "uniform":
        # Placeholder for the existing models
        for _ in range(num_total - num_models):
            index_list.append({})
        for _ in range(num_models):
            if data_type == "include":
                train_index = np.random.choice(
                    all_index_exclude_z,
                    int((configs["f_train"]) * dataset_size) - 1,
                    replace=False,
                )
                index_list.append(
                    {
                        "train": np.append(train_index, data_idx),
                        "test": all_index,
                        "audit": all_index,
                    }
                )
            elif data_type == "exclude":
                train_index = np.random.choice(
                    all_index_exclude_z,
                    int((configs["f_train"]) * dataset_size),
                    replace=False,
                )
                index_list.append(
                    {"train": train_index, "test": all_index, "audit": all_index}
                )
            else:
                raise ValueError(
                    f"{data_type} is not supported. Please use the value include or exclude to indicate whether you want to generate a set of dataset with or without the target point."
                )

    # We generate a list of dataset which is the same as the training dataset of the models indicated by the matched_in_idx but excluding the target point.
    elif configs["split_method"] == "leave_one_out":
        assert (
            matched_in_idx is not None
        ), "Please indicate the in-world model metdadata"
        assert (
            len(matched_in_idx) >= num_models
        ), "Input enough in-world (with the target point z) to generate the out world"

        index_list = []  # List of data split
        all_index = np.arange(N)
        all_index_exclude_z = np.array([i for i in all_index if i != data_idx])

        for _ in range(num_total - num_models):
            index_list.append({})
            associated_models.append(None)

        for metadata_idx in matched_in_idx:
            metadata = model_metadata_dict["model_metadata"][metadata_idx]
            train_index = np.delete(
                metadata["train_split"],
                np.where(metadata["train_split"] == data_idx)[0],
            )
            # Note: Since we are intested in the individual privacy risk, we consider the whole dataset as the test and audit dataset
            index_list.append(
                {
                    "train": train_index,
                    "test": [
                        i for i in all_index if i not in train_index and i != data_idx
                    ],
                    "audit": [
                        i for i in all_index if i not in train_index and i != data_idx
                    ],
                }
            )
            associated_models.append(metadata["model_idx"])

    else:
        raise ValueError(
            f"{configs['split_method']} is not supported. Please use uniform or leave_one_out splitting method."
        )

    dataset_splits = {
        "split": index_list,
        "split_method": configs["split_method"],
        "associated_models": associated_models,
    }
    return dataset_splits


def prepare_models(log_dir: dict, dataset, data_split, configs, model_metadata_dict):
    """Train models based on the dataset list

    Args:
        log_dir: Log directory that saved all the information, including the models.
        dataset: The whole dataset
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        configs (dict): Indicate the traininig information
        model_metadata_dict (dict): Metadata information about the existing models.
        matched_idx (List, optional): Index list of existing models that matchs configuration. Defaults to None.

    Returns:
        model_list: List of trained models
        model_metadata_dict: Updated Metadata of the existing models
        target_model_idx_list: Updated index list that matches the target model configurations.
    """
    # Initialize the model list
    model_list = []
    target_model_idx_list = []
    # Train the additional target models based on the dataset split
    for split in range(len(data_split)):
        meta_data = {}
        baseline_time = time.time()

        train_data = torch.utils.data.Subset(
            dataset, data_split["split"][split]["train"]
        )
        test_data = torch.utils.data.Subset(dataset, data_split["split"][split]["test"])
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=configs["batch_size"], shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=configs["test_batch_size"],
            shuffle=False,
            num_workers=2,
        )

        print(50 * "-")
        print(
            f"Training the {split}-th model: ",
            f"Train size {len(train_data)}, " f"Test size {len(test_data)}",
        )

        # Train the target model based on the configurations.
        model = get_model(configs["model_name"])
        model = train(model, train_loader, configs)
        # Test performance on the training dataset and test dataset
        test_loss, test_acc = inference(model, test_loader, configs["device"])
        train_loss, train_acc = inference(model, train_loader, configs["device"])
        model_list.append(copy.deepcopy(model))
        logging.info(
            "Prepare %s-th target model costs %s seconds ",
            split,
            time.time() - baseline_time,
        )
        print(f"Train accuracy {train_acc}, Train Loss {train_loss}")
        print(f"Test accuracy {test_acc}, Test Loss {test_loss}")
        print(50 * "-")

        # Update the model metadata and save the model
        model_idx = model_metadata_dict["current_idx"]
        model_metadata_dict["current_idx"] += 1
        with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
            pickle.dump(model.state_dict(), f)
        meta_data["train_split"] = data_split["split"][split]["train"]
        meta_data["test_split"] = data_split["split"][split]["test"]
        meta_data["audit_split"] = data_split["split"][split]["audit"]
        meta_data["num_train"] = len(data_split["split"][split]["train"])
        meta_data["optimizer"] = configs["optimizer"]
        meta_data["batch_size"] = configs["batch_size"]
        meta_data["epochs"] = configs["epochs"]
        meta_data["model_name"] = configs["model_name"]
        meta_data["split_method"] = data_split["split_method"]
        meta_data["model_idx"] = model_idx
        meta_data["learning_rate"] = configs["learning_rate"]
        meta_data["weight_decay"] = configs["weight_decay"]
        meta_data["model_path"] = f"{log_dir}/model_{model_idx}.pkl"
        # Check if there is any associated model (trained on the dataset differ by one record)
        if "associated_models" in data_split:
            if split < len(data_split["associated_models"]):
                meta_data["associated_models"] = {
                    "remove_from": data_split["associated_models"][split]
                }
        model_metadata_dict["model_metadata"][model_idx] = meta_data
        with open(f"{log_dir}/models_metadata.pkl", "wb") as f:
            pickle.dump(model_metadata_dict, f)
        target_model_idx_list.append(model_idx)
    return model_list, model_metadata_dict, target_model_idx_list


def get_info_source_population_attack(dataset, data_split, model, configs):
    """Prepare the information source for calling the core of privacy meter for the population attack

    Args:
        dataset: The whole dataset
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        model (model): Target Model.
        configs (dict): Auditing configuration

    Returns:
        target_dataset: List of target dataset on which we want to infer the membership
        audit_dataset:  List of auditing datasets we use for launch the attack
        target_model: List of target models we want to audit
        reference_model: List of reference models (which is the target model based on population attack)
    """
    train_data, train_targets = get_dataset_subset(dataset, data_split["train"])
    test_data, test_targets = get_dataset_subset(dataset, data_split["test"])
    audit_data, audit_targets = get_dataset_subset(dataset, data_split["audit"])
    target_dataset = Dataset(
        data_dict={
            "train": {"x": train_data, "y": train_targets},
            "test": {"x": test_data, "y": test_targets},
        },
        default_input="x",
        default_output="y",
    )

    audit_dataset = Dataset(
        data_dict={"train": {"x": audit_data, "y": audit_targets}},
        default_input="x",
        default_output="y",
    )
    target_model = PytorchModelTensor(
        model_obj=model,
        loss_fn=nn.CrossEntropyLoss(),
        device=configs["device"],
        batch_size=configs["audit_batch_size"],
    )
    return [target_dataset], [audit_dataset], [target_model], [target_model]


def get_info_source_reference_attack(
    log_dir,
    dataset,
    data_split,
    model,
    configs,
    model_metadata_dict,
    target_model_idx,
    # matched_reference_idx=None,
):
    """Prepare the information source for the reference attacks

     Args:
        log_dir: Log directory that saved all the information, including the models.
        dataset: The whole dataset.
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        model (model): Target Model.
        configs (dict): Auditing configuration.
        model_metadata_dict (dict): Model metedata dict.
        # matched_reference_idx (list, optional): List of existing reference models. Defaults to None.

    Returns:
        target_dataset: List of target dataset on which we want to infer the membership.
        audit_dataset:  List of auditing datasets we use for launch the attack (which is the target dataset based on reference attack)
        target_model: List of target models we want to audit.
        reference_model: List of reference models.
        model_metadata_dict: Updated metadata for the trained model.

    """

    # Construct the target dataset and target models
    train_data, train_targets = get_dataset_subset(dataset, data_split["train"])
    test_data, test_targets = get_dataset_subset(dataset, data_split["test"])
    target_dataset = Dataset(
        data_dict={
            "train": {"x": train_data, "y": train_targets},
            "test": {"x": test_data, "y": test_targets},
        },
        default_input="x",
        default_output="y",
    )
    target_model = PytorchModelTensor(
        model_obj=model,
        loss_fn=nn.CrossEntropyLoss(),
        device=configs["device"],
        batch_size=configs["audit_batch_size"],
    )

    # Construct reference models
    reference_models = []

    # Load existing reference models from disk using searching

    # Load existing reference models from disk
    if matched_reference_idx is not None:
        for metadata_idx in matched_reference_idx:
            metadata = model_metadata_dict["model_metadata"][metadata_idx]
            model = get_model(configs["model_name"])
            with open(f"{metadata['model_path']}", "rb") as f:
                model_weight = pickle.load(f)
            model.load_state_dict(model_weight)
            reference_models.append(
                PytorchModelTensor(
                    model_obj=copy.deepcopy(model),
                    loss_fn=nn.CrossEntropyLoss(),
                    device=configs["device"],
                    batch_size=configs["audit_batch_size"],
                )
            )

    print(f"Load existing {len(reference_models)} reference models")

    # Train additional reference models
    num_reference_models = configs["num_reference_models"] - len(reference_models)
    for reference_idx in range(num_reference_models):
        reference_data_idx = get_split(
            data_split["audit"],
            None,
            size=int(configs["f_reference_dataset"] * len(train_data)),
            split_method=configs["split_method"],
        )

        print(f"Training  {reference_idx}-th reference model")
        start_time = time.time()

        reference_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, reference_data_idx),
            batch_size=configs["batch_size"],
            shuffle=True,
            num_workers=2,
        )
        # reference_loader = torch.utils.data.DataLoader(get_cifar10_subset(
        #     dataset, reference_data_idx), batch_size=configs['batch_size'], shuffle=True, num_workers=2)
        reference_model = get_model(configs["model_name"])
        reference_model = train(reference_model, reference_loader, configs)
        # Test performance on the training dataset and test dataset
        train_loss, train_acc = inference(model, reference_loader, configs["device"])

        logging.info(
            f"Prepare {reference_idx}-th reference model costs {time.time()-start_time} seconds: Train accuracy (on auditing dataset) {train_acc}, Train Loss {train_loss}"
        )

        model_idx = model_metadata_dict["current_idx"]
        model_metadata_dict["current_idx"] += 1
        with open(f"{log_dir}/model_{model_idx}.pkl", "wb") as f:
            pickle.dump(reference_model.state_dict(), f)

        meta_data = {}
        meta_data["train_split"] = reference_data_idx
        meta_data["num_train"] = len(reference_data_idx)
        meta_data["optimizer"] = configs["optimizer"]
        meta_data["batch_size"] = configs["batch_size"]
        meta_data["epochs"] = configs["epochs"]
        meta_data["split_method"] = configs["split_method"]
        meta_data["model_idx"] = model_idx
        meta_data["learning_rate"] = configs["learning_rate"]
        meta_data["weight_decay"] = configs["weight_decay"]
        meta_data["model_name"] = configs["model_name"]
        meta_data["model_path"] = f"{log_dir}/model_{model_idx}.pkl"
        model_metadata_dict["model_metadata"][model_idx] = meta_data
        reference_models.append(
            PytorchModelTensor(
                model_obj=reference_model,
                loss_fn=nn.CrossEntropyLoss(),
                device=configs["device"],
                batch_size=configs["audit_batch_size"],
            )
        )

        # Save the updated metadata
        with open(f"{log_dir}/models_metadata.pkl", "wb") as f:
            pickle.dump(model_metadata_dict, f)

    return (
        [target_dataset],
        [target_dataset],
        [target_model],
        reference_models,
        model_metadata_dict,
    )


def prepare_information_source(
    log_dir,
    dataset,
    data_split,
    model_list,
    configs,
    model_metadata_dict,
    target_model_idx_list: List(int) = None,
    # matched_reference_idx=None,
):
    """Prepare the information source for calling the core of the privacy meter
    Args:
        log_dir: Log directory that saved all the information, including the models.
        dataset: The whole dataset
        data_split (dict): Data split information. 'split' contains a list of dict, each of which has the train, test and audit information. 'split_method' indicates the how the dataset is generated.
        model_list (List): List of target models.
        configs (dict): Auditing configuration.
        model_metadata_dict (dict): Model metedata dict.
        matched_reference_idx (list, optional): List of existing reference models. Defaults to None.

    Returns:
        target_info_source_list (List):
        reference_info_source_list (List):
        metric_list (List):
        log_dir_list (List):
        model_metadata_dict (dict): Updated metadata for the trained model.
    """
    reference_info_source_list = []
    target_info_source_list = []
    metric_list = []
    log_dir_list = []

    # Prepare the information source for each target model
    for split in range(len(model_list)):
        print(f"preparing information sources for {split}-th target model")
        if configs["algorithm"] == "population":
            (
                target_dataset,
                audit_dataset,
                target_model,
                audit_models,
            ) = get_info_source_population_attack(
                dataset, data_split["split"][split], model_list[split], configs
            )
            metrics = MetricEnum.POPULATION
        elif configs["algorithm"] == "reference":
            # Check if there are existing reference models
            (
                target_dataset,
                audit_dataset,
                target_model,
                audit_models,
                model_metadata_dict,
            ) = get_info_source_reference_attack(
                log_dir,
                dataset,
                data_split["split"][split],
                model_list[split],
                configs,
                model_metadata_dict,
                target_model_idx_list[split],
            )
            metrics = MetricEnum.REFERENCE
        metric_list.append(metrics)

        target_info_source = InformationSource(
            models=target_model, datasets=target_dataset
        )
        reference_info_source = InformationSource(
            models=audit_models, datasets=audit_dataset
        )
        reference_info_source_list.append(reference_info_source)
        target_info_source_list.append(target_info_source)

        # Save the log_dir for attacking different target model
        log_dir_path = f"{log_dir}/{configs['report_log']}/signal_{split}"
        Path(log_dir_path).mkdir(parents=True, exist_ok=True)
        log_dir_list.append(log_dir_path)

    return (
        target_info_source_list,
        reference_info_source_list,
        metric_list,
        log_dir_list,
        model_metadata_dict,
    )


def prepare_priavcy_risk_report(log_dir, audit_results, configs, save_path=None):
    """Generate privacy risk report based on the auditing report

    Args:
        log_dir: Log directory that saved all the information, including the models.
        audit_results: Privacy meter results.
        configs (dict): Auditing configuration.
        save_path (str, optional): Report path. Defaults to None.

    Raises:
        NotImplementedError: Check if the report for the privacy game is implemented.

    """
    audit_report.REPORT_FILES_DIR = "privacy_meter/report_files"
    if save_path is None:
        save_path = log_dir

    if configs["privacy_game"] in [
        "privacy_loss_model",
        "avg_privacy_loss_training_algo",
    ]:
        # Generate privacy risk report for auditing the model
        if len(audit_results) == 1 and configs["privacy_game"] == "privacy_loss_model":
            ROCCurveReport.generate_report(
                metric_result=audit_results[0],
                inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                save=True,
                filename=f"{save_path}/ROC.png",
            )
            SignalHistogramReport.generate_report(
                metric_result=audit_results[0][0],
                inference_game_type=InferenceGame.PRIVACY_LOSS_MODEL,
                save=True,
                filename=f"{save_path}/Histogram.png",
            )
        # Generate privacy risk report for auditing the training algorithm
        elif (
            len(audit_results) > 1
            and configs["privacy_game"] == "avg_privacy_loss_training_algo"
        ):
            ROCCurveReport.generate_report(
                metric_result=audit_results,
                inference_game_type=InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO,
                save=True,
                filename=f"{save_path}/ROC.png",
            )

            SignalHistogramReport.generate_report(
                metric_result=audit_results,
                inference_game_type=InferenceGame.AVG_PRIVACY_LOSS_TRAINING_ALGO,
                save=True,
                filename=f"{save_path}/Histogram.png",
            )
        else:
            raise ValueError(
                f"{len(audit_results)} results are not enough for {configs['privacy_game']})"
            )
    else:
        raise NotImplementedError(f"{configs['privacy_game']} is not implemented yet")
