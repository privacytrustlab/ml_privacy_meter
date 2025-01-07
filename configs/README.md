# Configs
Here, we explain each field in the config files.
- **run**: Configurations related to this specific run 
  - **random_seed**: integer number of specifying random seed. Each run of experiments will use the same random seed.
  - **log_dir**: Path to where all the information will be saved, including models and computed signals. If the directory contains models, these models will be loaded instead of trained. Hence, to run experiments with new models, we need to change the log_dir.
  - **time_log**: Indicate whether to log the time for each step. If `True`, a time log will be saved
  - **num_experiments**: Number of target models we attack. If it is more than 1, an aggregate report will be generated in the end

- **audit**: Configurations related to auditing
  - **privacy_game**: Indicate the type of privacy game/notion. We currently support the `privacy_loss_model` game. We will add more games in the future.
  - **algorithm**: The membership inference attack used for auditing. We currently support the RMIA introduced by Zarifzadeh et al. 2024(https://openreview.net/pdf?id=sT7UJh5CTc) and the LOSS attack
  - **num_ref_models**: Number of reference models used to audit each target model
  - **device**: The device we want to use for inferring signals and auditing models
  - **report_log**: The folder name where we save the log and auditing report
  - **batch_size**: Batch size for evaluating models and inferring signals.
  - **data_size**: The size of the dataset in auditing. If not specified, the entire dataset is used. Must be an even number. The sampled auditing dataset will contain equal numbers of IN and OUT data samples according to the membership information from the first target model.

- **train**: Configuration related to training
  - **model_name**: The model type. We support CNN, wrn28-1, wrn28-2, wrn28-10, vgg16, mlp, gpt2 and speedyresnet. More model types can be added in `/models/`.
  - **tokenizer**: The tokenizer type. It can be any tokenizer or local checkpoint supported by the `transformers` library. For non-text datasets, this field can be dropped.
  - **device**: The device we want to use for training models. Note for `transformers`, the behavior from Huggingface's `Trainer` class is to use all GPUs available.
  - **batch_size**: Batch size for training models.
  - **learning_rate**: Learning rate for training models.
  - **weight_decay**: Weight decay for training models.
  - **epochs**: Number of epochs for training models.
  - **optimizer**: Optimizer for training models. We support `SGD`, `Adam`, `AdamW`. More optimizers can be added in `get_optimizer` in `trainers/default_trainer.py`.
  - **peft**: Configuration related to peft. It can be dropped if not needed.
- **data**: Configuration related to datasets
  - **dataset**: The name of the dataset. We support cifar10, cifar100, purchase100 and texas100 and agnews by default.
  - **data_dir**: The directory where the dataset is stored. If the dataset is not found in the directory, it will be downloaded.
  - **tokenize**: Indicate whether to tokenize the dataset. If `True`, the dataset will be tokenized using the tokenizer specified in the next field. It can be dropped if not needed.
  - **tokenizer**: The tokenizer type. It can be any tokenizer or local checkpoint supported by the `transformers` library. For non-text datasets, this field can be dropped.
- **dp_audit**: Configuration related to dp auditing
  - **canary_dataset**: the name of the canary dataset. We support cifar10_canary constructed by randomly labelled images from CIFAR10 dataset
  - **canary_size**: the number of data records to load from the canary dataset
  - **training_alg**: whether to use `dp` or `nondp` training algorithm