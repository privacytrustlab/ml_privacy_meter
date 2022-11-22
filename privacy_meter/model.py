from abc import ABC, abstractmethod
from copy import deepcopy
from multiprocessing import reduction

import numpy as np
import torch
from opacus import GradSampleModule # For speeding up the gradient computation
########################################################################################################################
# MODEL CLASS
########################################################################################################################


class Model(ABC):
    """
    Interface to query a model without any assumption on how it is implemented.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
        """
        self.model_obj = model_obj
        self.loss_fn = loss_fn

    @abstractmethod
    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output
        """
        pass

    @abstractmethod
    def get_loss(self, batch_samples, batch_labels, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        pass

    @abstractmethod
    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        pass

    @abstractmethod
    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        pass


########################################################################################################################
# PYTORCH_MODEL CLASS
########################################################################################################################


class PytorchModel(Model):
    """
    Inherits from the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with pytorch models.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
        """

        # Imports torch with global scope
        globals()['torch'] = __import__('torch')

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Add hooks to the layers (to access their value during a forward pass)
        self.intermediate_outputs = {}
        for (i, l) in enumerate(list(self.model_obj._modules.keys())):
            getattr(self.model_obj, l).register_forward_hook(self.__forward_hook(l))

        # Create a second loss function, per point
        self.loss_fn_no_reduction = deepcopy(loss_fn)
        self.loss_fn_no_reduction.reduction = 'none'

    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output.
        """
        return self.model_obj(torch.Tensor(batch_samples)).detach().numpy()

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        if per_point:
            return self.loss_fn_no_reduction(
                self.model_obj(torch.Tensor(batch_samples)),
                torch.Tensor(batch_labels)
            ).detach().numpy()
        else:
            return self.loss_fn(self.model_obj(torch.Tensor(batch_samples)), torch.Tensor(batch_labels)).item()

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        loss = self.loss_fn(self.model_obj(torch.Tensor(batch_samples)), torch.Tensor(batch_labels))
        loss.backward()
        return [p.grad.numpy() for p in self.model_obj.parameters()]

    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        if forward_pass:
            _ = self.get_logits(torch.Tensor(batch_samples))
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
            layer_name: Name of the layer to access.

        Returns:
            A hook to be registered using register_forward_hook.
        """

        def hook(module, input, output):
            self.intermediate_outputs[layer_name] = output

        return hook

########################################################################################################################
# TENSORFLOW_MODEL CLASS
########################################################################################################################


class TensorflowModel(Model):
    """Inherits from the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with tensorflow models.
    """

    def __init__(self, model_obj, loss_fn):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
        """

        # Imports tensorflow with global scope
        globals()['tf'] = __import__('tensorflow')

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Store the layers names in a dict (to access intermediate outputs by names)
        self.layers_names = dict(zip(
            [layer._name for layer in self.model_obj.layers],
            [i for i in range(len(self.model_obj.layers))]
        ))

        # Create a second loss function, per point
        self.loss_fn_no_reduction = deepcopy(loss_fn)
        self.loss_fn_no_reduction.reduction = 'none'

    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output.
        """
        return self.model_obj(batch_samples).numpy()

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        if per_point:
            return self.loss_fn_no_reduction(batch_labels, self.get_logits(batch_samples)).numpy()
        else:
            return self.loss_fn(batch_labels, self.get_logits(batch_samples)).numpy()

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

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
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
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
            x: List of tf tensors.

        Returns:
            A list of numpy arrays.
        """
        if isinstance(x, list):
            return [self.__tf_list_to_np_list(y) for y in x]
        else:
            return x.numpy()

########################################################################################################################
# LANGUAGE_MODEL CLASS
########################################################################################################################


class LanguageModel(Model):
    """
    Inherits from the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular abstract class is to be used with language models.
    """

    @abstractmethod
    def get_perplexity(self, batch_samples):
        """Function to get the perplexity of the model loss, on a given input sequence.

        Args:
            batch_samples: Model input.

        Returns:
            A list of perplexity values.
        """
        pass

########################################################################################################################
# HUGGINGFACE_CAUSAL_LANGUAGE_MODEL_CLASS
########################################################################################################################


class HuggingFaceCausalLanguageModel(LanguageModel):
    """
    Inherits from the LanguageModel class, an interface to query a language model without any assumption on how it is
    implemented.
    This particular class is to be used with HuggingFace causal language models.
    """

    def __init__(self, model_obj, loss_fn, stride=64):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
            stride: Window size that will be used by the fixed length causal model for processing an input sequence.
        """

        # Imports torch with global scope
        globals()['torch'] = __import__('torch')

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        self.stride = stride

        # Create a second loss function, per point
        self.loss_fn_no_reduction = deepcopy(loss_fn)
        self.loss_fn_no_reduction.reduction = 'none'

    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output.
        """
        batch_samples = torch.tensor(batch_samples, dtype=torch.long)
        batch_labels = batch_samples.clone()
        with torch.no_grad():
            outputs = self.model_obj(batch_samples, labels=batch_labels)
        return outputs.get("logits")

    def get_loss(self, batch_samples, batch_labels=None, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        pass

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        pass

    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        pass

    def get_perplexity(self, batch_samples):
        """Function to get the perplexity of the model loss, on a given input sequence.

        Args:
            batch_samples: Model input.

        Returns:
            A list of perplexity values.
        """
        max_length = self.model_obj.config.n_positions

        ppl_values = []
        for sample in batch_samples:
            sample_length = len(sample)

            sample = np.expand_dims(sample, axis=0)  # the model takes in a batch of sequences
            sample = torch.tensor(sample, dtype=torch.long)

            nlls = []
            for i in range(0, sample_length, self.stride):
                begin_loc = max(i + self.stride - max_length, 0)
                end_loc = min(i + self.stride, sample_length)

                trg_len = end_loc - i  # may be different from stride on last loop

                input_ids = sample[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model_obj(input_ids, labels=target_ids)
                    neg_log_likelihood = outputs[0] * trg_len

                nlls.append(neg_log_likelihood)
            ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

            ppl_values.append(ppl)

        return ppl_values



class PytorchModelTensor(Model):
    """
    Inherits from the Model class, an interface to query a model without any assumption on how it is implemented.
    This particular class is to be used with pytorch models.
    """

    def __init__(self, model_obj, loss_fn,device='cpu',batch_size=25):
        """Constructor

        Args:
            model_obj: Model object.
            loss_fn: Loss function.
            device: Indicate the device to compute the signals
            batch_size: Indicate the batch size to compute the signals
        """

        # Imports torch with global scope
        globals()['torch'] = __import__('torch')

        # Initializes the parent model
        super().__init__(model_obj, loss_fn)

        # Create a second loss function, per point
        self.loss_fn_no_reduction = deepcopy(loss_fn)
        self.loss_fn_no_reduction.reduction = 'none'
        self.device = device
        self.grad_sampler_model = None
        self.model_obj.to(device)
        self.batch_size = batch_size
        

    def get_logits(self, batch_samples):
        """Function to get the model output from a given input.

        Args:
            batch_samples: Model input.

        Returns:
            Model output.
        """

        return self.model_obj(batch_samples).detach().numpy()

    def get_loss(self, batch_samples, batch_labels, per_point=True):
        """Function to get the model loss on a given input and an expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.
            per_point: Boolean indicating if loss should be returned per point or reduced.

        Returns:
            The loss value, as defined by the loss_fn attribute.
        """
        if per_point:
            self.model_obj.eval()
            loss_list = []
            batched_samples = torch.split(batch_samples,self.batch_size)
            batched_labels = torch.split(batch_labels,self.batch_size)
            for x, y in zip(batched_samples,batched_labels):
                x=x.to(self.device)
                y = y.to(self.device)
                loss = self.loss_fn_no_reduction(self.model_obj(x), y)
                loss_list.append(loss.detach()) # to avoid the OOM
            return torch.cat(loss_list).detach().cpu().numpy()
        else:
            return self.loss_fn(self.model_obj(torch.Tensor(batch_samples)), torch.Tensor(batch_labels)).item()

    def get_grad(self, batch_samples, batch_labels):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        if self.grad_sampler_model is None:
            self.grad_sampler_model = GradSampleModule(self.model_obj)
        
        self.grad_sampler_model._module.train()
        batch_size = 25
        grad_norm = []
        batched_samples = torch.split(batch_samples,batch_size)
        batched_labels = torch.split(batch_labels,batch_size)
        self.grad_sampler_model.zero_grad()
        for x, y in zip(batched_samples,batched_labels):
            batch_samples = batch_samples.to(self.device)
            batched_labels = batched_labels.to(self.device)
            loss = self.loss_fn(self.grad_sampler_model(x), y).sum()
            loss.backward()
            
        grade = torch.cat([p.grad_sample.view(-1) for p in self.grad_sampler_model.parameters()]).detach().cpu().numpy()
        return grade

    
    def get_gradnorm(self, batch_samples, batch_labels,is_features=True,layer_number=1):
        """Function to get the gradient of the model loss with respect to the model parameters, on a given input and an
        expected output.

        Args:
            batch_samples: Model input.
            batch_labels: Model expected output.

        Returns:
            A list of gradients of the model loss (one item per layer) with respect to the model parameters.
        """
        if self.grad_sampler_model is None:
            self.grad_sampler_model = GradSampleModule(self.model_obj)
        self.grad_sampler_model._module.train()
        grad_norm = []
        batched_samples = torch.split(batch_samples,self.batch_size)
        batched_labels = torch.split(batch_labels,self.batch_size)
        self.grad_sampler_model.zero_grad()
        for x, y in zip(batched_samples,batched_labels):
            x = x.to(self.device)
            y = y.to(self.device)
            loss = self.loss_fn(self.grad_sampler_model(x), y).sum()
            loss.backward()
            if is_features:
                grad_norm.append(torch.norm(self.grad_sampler_model.features[layer_number].weight.grad_sample.reshape(self.batch_size,-1).detach(),dim=1)) # this detach can not be deleted
            else:
                grad_norm.append(torch.norm(self.grad_sampler_model.classifier[layer_number].weight.grad_sample.reshape(self.batch_size,-1).detach(),dim=1)) # this detach can not be deleted

            self.grad_sampler_model.zero_grad()
        grad_norm = torch.cat(grad_norm).detach().cpu().numpy()
        return grad_norm

    def get_intermediate_outputs(self, layers, batch_samples, forward_pass=True):
        """Function to get the intermediate output of layers (a.k.a. features), on a given input.

        Args:
            layers: List of integers and/or strings, indicating which layers values should be returned.
            batch_samples: Model input.
            forward_pass: Boolean indicating if a new forward pass should be executed. If True, then a forward pass is
                executed on batch_samples. Else, the result is the one of the last forward pass.

        Returns:
            A list of intermediate outputs of layers.
        """
        if forward_pass:
            _ = self.get_logits(torch.Tensor(batch_samples))
        layer_names = []
        for layer in layers:
            if isinstance(layer, str):
                layer_names.append(layer)
            else:
                layer_names.append(list(self.model_obj._modules.keys())[layer])
        return [self.intermediate_outputs[layer_name].detach().numpy() for layer_name in layer_names]
