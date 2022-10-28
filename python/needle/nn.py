"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).reshape((1, -1)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        if self.bias is not None:
            return X.matmul(self.weight) + self.bias.broadcast_to((X.shape[0], self.out_features))
        else:
            return X.matmul(self.weight)

class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ### END YOUR SOLUTION
        return X.reshape((X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)



class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        # logits [B, K]
        # y [B]
        B = logits.shape[0]
        K = logits.shape[1]
        Z = ops.LogSumExp(axes=1)(logits)
        # print("Z dtype", Z.dtype)
        # print("OK", ops.log(ops.summation(ops.exp(logits), axes=1)).detach())
        y_one_hot = init.one_hot(K, y)
        res = (Z - ops.summation(logits * y_one_hot, axes=1)).sum() / Tensor(B, dtype="float32")
        # print("v1", (logits * y_one_hot).dtype)
        # print("v2", ops.summation(logits * y_one_hot, axes=1).dtype)
        # print("v3", (Z - ops.summation(logits * y_one_hot, axes=1)).dtype)
        # print("v4", (Z - ops.summation(logits * y_one_hot, axes=1)).sum().dtype)
        # print("B", B.dtype)
        # print("res dtype", res.dtype)
        return res



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(self.dim, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True))
        self.running_mean = Parameter(init.zeros(self.dim, requires_grad=False))
        self.running_var = Parameter(init.ones(self.dim, requires_grad=False))

    def forward(self, x: Tensor) -> Tensor:

        """
        E_x = ops.broadcast_to((x.sum(axes=1) / x.shape[1]).reshape((-1, 1)), x.shape)
        x_fixed = x - E_x
        VAR_x = ops.broadcast_to((x_fixed * x_fixed / x.shape[1]).sum(axes=1).reshape((-1, 1)), x.shape)
        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        return w * x_fixed / (VAR_x + self.eps) ** 0.5 + b
        """
        mu_raw = x.sum(axes=0) / x.shape[0]
        mu = ops.broadcast_to(mu_raw.reshape((1, -1)), x.shape)
        x_fixed = x - mu
        var_raw = (x_fixed * x_fixed / x.shape[0]).sum(axes=0)
        var = ops.broadcast_to(var_raw.reshape((1, -1)), x.shape)
        # print("x", x.shape, x)
        # print("mu_raw", mu_raw.shape, mu_raw)
        # print("mu", mu.shape, mu)
        # print("var_raw", var_raw.shape, var_raw)
        # print("var", var.shape, var)
        w = ops.broadcast_to(self.weight.reshape((1, -1)), x.shape)
        b = ops.broadcast_to(self.bias.reshape((1, -1)), x.shape)
        # print("w", w.shape, w)
        # print("b", b.shape, b)
        if self.training:
            self.running_mean = (1 - self.momentum) * self.running_mean.detach() + self.momentum * mu_raw
            self.running_var = (1 - self.momentum) * self.running_var.detach() + self.momentum * var_raw
            return w * (x - mu) / (var + self.eps) ** 0.5 + b
        else:
            return w * (x - self.running_mean.detach().reshape((1, -1)).broadcast_to(x.shape)) / (self.running_var.detach().reshape((1, -1)).broadcast_to(x.shape) + self.eps) ** 0.5 + b




class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # print("x", x.shape, x)
        E_x = ops.broadcast_to((x.sum(axes=1) / x.shape[1]).reshape((-1, 1)), x.shape)
        # print("E_x", E_x.shape, E_x)
        x_fixed = x - E_x
        # print("x_fixed", x_fixed.shape, x_fixed)
        VAR_x = ops.broadcast_to((x_fixed * x_fixed / x.shape[1]).sum(axes=1).reshape((-1, 1)), x.shape)
        # print("VAR_x", VAR_x.shape, VAR_x)
        # print("weight", self.weight.shape, self.weight)
        w = ops.broadcast_to(self.weight, x.shape)
        b = ops.broadcast_to(self.bias, x.shape)
        # print("w", w.shape, w)
        return w * x_fixed / (VAR_x + self.eps) ** 0.5 + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



