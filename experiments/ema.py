"""Pytorch implementation of EMA"""
import copy
from collections import OrderedDict
from copy import deepcopy
from sys import stderr

import torch
from torch import Tensor, nn


class EMA(nn.Module):
    # https://www.zijianhu.com/post/pytorch/ema/
    # https://objax.readthedocs.io/en/latest/_modules/objax/optimizer/ema.html#ExponentialMovingAverage
    def __init__(self, model: nn.Module, decay: float, debias: bool = True):
        super().__init__()
        self.decay = decay
        self.model = model
        self.eps = 1e-6
        self.debias = debias
        self.shadow = deepcopy(self.model)
        self.step = 0
        for param in self.shadow.parameters():
            if self.debias:
                param.data.zero_()
            param.detach_()

    @torch.no_grad()
    def update(self):
        self.step += 1
        if not self.training:
            print(
                "EMA update should only be called during training",
                file=stderr,
                flush=True,
            )
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())
        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            new_param = shadow_params[name] * self.decay + (1.0 - self.decay) * param
            shadow_params[name].copy_(new_param)

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()
        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            return self.model(inputs)
        else:
            # update the parameter when calling the model
            pre_model_params = copy.deepcopy(self.shadow.state_dict())
            shadow_params = OrderedDict(self.shadow.named_parameters())
            if self.debias:
                # print(shadow_params['linear.bias'],self.model.state_dict()['linear.bias'])
                step = self.step
                debias = 1 / (1 - (1 - self.eps) * self.decay**step)
                for name, param in shadow_params.items():
                    shadow_params[name].copy_(param * debias)
                output = self.shadow(inputs)
                for name, param in shadow_params.items():
                    shadow_params[name].copy_(pre_model_params[name])
                return output
            else:
                return self.shadow(inputs)
