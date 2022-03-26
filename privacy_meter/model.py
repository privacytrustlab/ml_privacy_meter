from abc import ABC, abstractmethod
import warnings

try:
    import tensorflow as tf
except ImportError:
    warnings.warn("tensorflow package not found: TensorflowModel might not work.", category=ImportWarning)

try:
    from torch import Tensor
except ImportError:
    warnings.warn("torch package not found: PytorchModel might not work.", category=ImportWarning)


class Model(ABC):
    """
    Interface to query a model without any assumption on how it is implemented.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: model object
            loss_fn: loss function
        """
        self.model_obj = model_obj
        self.loss_fn = loss_fn

    @abstractmethod
    def get_outputs(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input

        Returns:
            Model output
        """
        pass

    @abstractmethod
    def get_loss(self, batch_samples, batch_labels):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input
            batch_labels: Model expected output

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        pass

    @abstractmethod
    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input
            batch_labels: Model expected output

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        pass

    @abstractmethod
    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned
            batch_samples: Model input
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        pass


class PytorchModel(Model):
    """
    Inherits of the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with pytorch models.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: model object
            loss_fn: loss function
        """

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Add hooks to the layers (to access their value during a forward pass)
        self.intermediate_outputs = {}
        for (i, l) in enumerate(list(self.model_obj._modules.keys())):
            getattr(self.model_obj, l).register_forward_hook(self.__forward_hook(l))

    def get_outputs(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input

        Returns:
            Model output
        """
        return self.model_obj(Tensor(batch_samples)).detach().numpy()

    def get_loss(self, batch_samples, batch_labels):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input
            batch_labels: Model expected output

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        return self.loss_fn(self.model_obj(Tensor(batch_samples)), Tensor(batch_labels)).item()

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input
            batch_labels: Model expected output

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        loss = self.loss_fn(self.model_obj(Tensor(batch_samples)), Tensor(batch_labels))
        loss.backward()
        return [p.grad.numpy() for p in self.model_obj.parameters()]

    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned
            batch_samples: Model input
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        if forward_pass:
            _ = self.get_outputs(Tensor(batch_samples))
        layer_names = []
        for layer in layers:
            if isinstance(layer, str):
                layer_names.append(layer)
            else:
                layer_names.append(list(self.model_obj._modules.keys())[layer])
        return [self.intermediate_outputs[layer_name].detach().numpy() for layer_name in layer_names]

    def __forward_hook(self, layer_name):
        """Private helper function to access outputs of intermediate layers.

        Args:
            layer_name: Name of the layer to access

        Returns:
            A hook to be registered using register_forward_hook.
        """

        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output

        return hook


class TensorflowModel(Model):
    """Inherits of the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with tensorflow models.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: model object
            loss_fn: loss function
        """

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Store the layers names in a dict (to access intermediate outputs by names)
        self.layers_names = dict(zip(
            [layer._name for layer in self.model_obj.layers],
            [i for i in range(len(self.model_obj.layers))]
        ))

    def get_outputs(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input

        Returns:
            Model output
        """
        return self.model_obj(batch_samples).numpy()

    def get_loss(self, batch_samples, batch_labels):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input
            batch_labels: Model expected output

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        return self.loss_fn(self.get_outputs(batch_samples), batch_labels).numpy()

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input
            batch_labels: Model expected output

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        with tf.GradientTape() as t:
            loss = self.loss_fn(self.model_obj(batch_samples), batch_labels)
        grad = t.gradient(loss, [self.model_obj.weights])
        return self.__tf_list_to_np_list(grad)

    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned
            batch_samples: Model input
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
            executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        assert forward_pass, 'Implementation of get_intermediate_outputs in TensorflowModel requires forward_pass=True'

        extractor = tf.keras.Model(
            inputs=self.model_obj.inputs,
            outputs=[layer.output for layer in self.model_obj.layers]
        )
        layers_indices = []
        for layer in layers:
            if isinstance(layer, str):
                layers_indices.append(self.layers_names[layer])
            else:
                layers_indices.append(layer)
        features = extractor(batch_samples)
        features = [features[i] for i in layers_indices]
        return self.__tf_list_to_np_list(features)

    def __tf_list_to_np_list(self, x):
        """Private helper function to recursively convert lists of tf tensors to lists of numpy arrays.

        Args:
            x: List of tf tensors

        Returns:
            A list of numpy arrays
        """
        if isinstance(x, list):
            return [self.__tf_list_to_np_list(y) for y in x]
        else:
            return x.numpy()
