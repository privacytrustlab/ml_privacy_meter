from typing import Tuple, Dict

from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def create_training_args(configs: Dict) -> TrainingArguments:
    """Creates and returns the training arguments for the transformer model."""
    return TrainingArguments(
        output_dir=configs["run"]["log_dir"],
        num_train_epochs=configs["train"]["epochs"],
        per_device_train_batch_size=configs["train"]["batch_size"],
        per_device_eval_batch_size=configs["train"]["batch_size"],
        warmup_steps=500,
        optim=configs["train"]["optimizer"],
        weight_decay=configs["train"]["weight_decay"],
        learning_rate=configs["train"]["learning_rate"],
        save_strategy="no",
        logging_strategy="epoch",
        eval_strategy="epoch",
    )


def setup_tokenizer(configs: Dict) -> AutoTokenizer:
    """Loads the tokenizer and ensures pad token is set."""
    tokenizer = AutoTokenizer.from_pretrained(
        configs["data"]["tokenizer"], clean_up_tokenization_spaces=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(
            "The tokenizer pad token is None. Setting it to the EOS token for padding. "
            "If this is not desired, please set the pad token manually."
        )
    return tokenizer


def train_transformer(
    trainset, model: PreTrainedModel, configs: Dict, testset
) -> Tuple[PreTrainedModel, float, float]:
    """Train a Hugging Face transformer model without any PEFT (LoRA) modifications."""
    if not isinstance(model, PreTrainedModel):
        raise ValueError("The provided model is not a Hugging Face transformer model")

    training_args = create_training_args(configs)
    tokenizer = setup_tokenizer(configs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
    )

    trainer.train()
    train_loss = trainer.state.log_history[-1]["train_loss"]
    test_loss = trainer.state.log_history[-2]["eval_loss"]

    return model, train_loss, test_loss


def get_peft_model_config(configs: Dict) -> LoraConfig:
    """Get the PEFT model configuration."""
    if "peft" not in configs["train"]:
        raise ValueError("LoRA configuration is not provided in the configuration file")

    if configs["train"]["peft"]["type"] == "lora":
        return LoraConfig(
            fan_in_fan_out=configs["train"]["peft"]["fan_in_fan_out"],
            inference_mode=False,
            r=configs["train"]["peft"]["r"],
            target_modules=configs["train"]["peft"]["target_modules"],
            lora_alpha=32,
            lora_dropout=0.05,
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        raise NotImplementedError("Only LoRA is supported in this implementation.")


def train_transformer_with_peft(
    trainset, model: PreTrainedModel, configs: Dict, testset
) -> Tuple[PreTrainedModel, float, float]:
    """Train a Hugging Face transformer model with PEFT (LoRA) modifications."""
    if not isinstance(model, PreTrainedModel):
        raise ValueError("The provided model is not a Hugging Face transformer model")

    # Apply PEFT (LoRA) configuration
    peft_config = get_peft_model_config(configs)
    peft_model = get_peft_model(model, peft_config)

    training_args = create_training_args(configs)
    tokenizer = setup_tokenizer(configs)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=trainset,
        eval_dataset=testset,
        tokenizer=tokenizer,
    )

    trainer.train()
    train_loss = trainer.state.log_history[-1]["train_loss"]
    test_loss = trainer.state.log_history[-2]["eval_loss"]

    return peft_model, train_loss, test_loss
