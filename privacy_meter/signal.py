from .model import Model
from .dataset import Dataset

def loss_signal_func(model: Model, dataset: Dataset,
                     split_name,
                     input_feature_name='<default_input>',
                     output_feature_name='<default_output>',
                     indices=None):
    """
    Function to return the loss values over the specified split of the
    provided dataset using the provided model.
    Args:
        model: Model used for computing loss values
        dataset: Dataset which will be used for providing input to the model
        split_name: Name of the split
        input_feature_name: Name of the input feature
        output_feature_name: Name of output feature e.g. true labels
        indices: Optional list of indices. If not specified, the entire subset is returned.
    Returns:
        loss_values: Sequence of loss values computed using the provided model and dataset
    """
    print(f"Getting loss values from {split_name} split of the dataset...")

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                      reduction=tf.keras.losses.Reduction.NONE)

    y_true = dataset.get_feature(split_name=split_name, feature_name=output_feature_name, indices=indices)

    x = dataset.get_feature(split_name=split_name, feature_name=input_feature_name, indices=indices)
    y_pred = model.get_outputs(batch_samples=x)

    loss_values = loss_fn(y_true=y_true, y_pred=y_pred)

    return loss_values
