"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp, as_tuple
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        return (self.scalar * power_scalar(node.inputs[0], self.scalar - 1) * out_grad, )


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return array_api.divide(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return (out_grad / rhs, -1 * out_grad * lhs / power_scalar(rhs, 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is not None:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            dim = len(a.shape)
            return array_api.swapaxes(a, dim -2, dim -1)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # print("reshape forward  input:\t", a.shape, a, "\tout shape", self.shape)
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        # todo reverse reshape
        # print("reshape backward input:\t", node.inputs[0].shape, node.inputs[0])
        # print("reshape backward output", node.shape, node)
        # print("reshape backward out_grad", out_grad.shape, out_grad)
        return reshape(out_grad, node.inputs[0].shape)
        # return Tensor(array_api.reshape(out_grad, node.inputs[0].shape))


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # (2,3) => (4,2,3) not ok for (2, 3, 4)
        shape_in = node.inputs[0].shape
        shape_out = out_grad.shape
        # 只能在最前面加维度，后面只能做1->x的提升
        # 分两步，一步是新增维度做sum，去除axis
        # 第二步是保留dim的sum
        # print("shape_in:\t", shape_in)
        # print("shape_out:\t", shape_out)
        if len(shape_in) != len(shape_out):
            out_grad = summation(out_grad, tuple(i for i in range(len(shape_out) - len(shape_in))))
        axes = []
        for i, dim in enumerate(shape_in):
            if dim == 1:
                axes.append(i)
        # print("axes:\t", axes)
        return summation(out_grad, tuple(axes)).reshape(shape_in)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)


    def gradient(self, out_grad, node):
        shape_in = node.inputs[0].shape
        shape_out = out_grad.shape
        shape_tmp = list(shape_in)
        if self.axes is not None:
            for ax in as_tuple(self.axes):
                shape_tmp[ax] = 1
        else:
            shape_tmp = [1 for _ in shape_in]
        # print("shape_in:\t", shape_in)
        # print("shape_out:\t", shape_out)
        # print("shape_tmp:\t", shape_tmp)
        return broadcast_to(out_grad.reshape(shape_tmp), shape_in)



def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        # m x n, n x k  ==> m x k
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = matmul(transpose(lhs), out_grad)
        if len(lhs_grad.shape) != len(lhs.shape):
            lhs_grad = summation(lhs_grad, axes=tuple(i for i in range(len(lhs_grad.shape) - len(lhs.shape))))
        if len(rhs_grad.shape) != len(rhs.shape):
            rhs_grad = summation(rhs_grad, axes=tuple(i for i in range(len(rhs_grad.shape) - len(rhs.shape))))
        return lhs_grad, rhs_grad
        # b1 x b2 x m x n, b1 x b2 x n x k => b1 x b2 x m x k
        # b1 x b2 x m x n, n x k => b1 x b2 x m x k


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)



class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, 0)

    def gradient(self, out_grad, node):
        return Tensor(array_api.where(node.inputs[0].realize_cached_data() > 0, out_grad.realize_cached_data(), 0))


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        return array_api.log(
            array_api.sum(array_api.exp(
                Z - array_api.max(Z, axis=self.axes, keepdims=True)), axis=self.axes)) \
                + array_api.max(Z, axis=self.axes)

    def gradient(self, out_grad, node):
        # print("out_grad", out_grad.shape, out_grad)
        # print("node", node.shape, node)
        Z = node.inputs[0].realize_cached_data()
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        Z_fixed = Z - Z_max
        Z_exp = array_api.exp(Z_fixed)
        Z_exp_sum = array_api.sum(Z_exp, axis=self.axes, keepdims=True)
        # print("axes", self.axes)
        # print("Z", Z.shape, Z)
        # print("Z_max", Z_max.shape, Z_max)
        # print("Z_fixed", Z_fixed.shape, Z_fixed)
        # print("Z_exp", Z_exp.shape, Z_exp)
        # print("Z_exp_sum", Z_exp_sum.shape, Z_exp_sum)
        # print("out_grad", out_grad.shape, out_grad)

        shape_in = node.inputs[0].shape
        shape_tmp = list(shape_in)
        if self.axes is not None:
            for ax in as_tuple(self.axes):
                shape_tmp[ax] = 1
        else:
            shape_tmp = [1 for _ in shape_in]
        # print("shape_tmp", shape_tmp)
        return broadcast_to(out_grad.reshape(shape_tmp), shape_in) * Tensor(Z_exp/Z_exp_sum)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)