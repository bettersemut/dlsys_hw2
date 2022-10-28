"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # print("in", id(self.params[0]))
        # print("grad", self.params[0].grad.detach())
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if self.momentum != 0:
                if i not in self.u:
                    self.u[i] = ndl.init.zeros(*p.shape)
                self.u[i] = self.momentum * self.u[i] + (1 - self.momentum) * (p.grad.detach() + self.weight_decay * p.detach())
                p.data = p - self.lr * self.u[i]
            else:
                # if self.params[i].grad is None:
                #     print("p", self.params[i].shape, self.params[i])
                # if i == 0:
                #     print("inner id before", id(p))
                #     # print("p before", p.shape, p)
                #     prev = p.numpy()
                #     print("p dtype", p.dtype)
                #     print("grad dtype", p.grad.detach().dtype)

                # print("grad", p.grad.shape, p.grad)
                p.data = p - self.lr * p.grad.detach() - self.lr * self.weight_decay * p.detach()
                # if i == 0:
                #     print("inner id after", id(p))
                #     # print("p after", p.shape, p)
                #     print("p l1 dis", np.abs(p.numpy() - prev).max())

        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            if i not in self.m:
                self.m[i] = np.zeros(p.shape)
            if i not in self.v:
                self.v[i] = np.zeros(p.shape)
            # grad = p.grad.detach()
            grad = p.grad.numpy() + self.weight_decay * p.numpy()
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_t = (self.m[i] / (1 - self.beta1 ** self.t))
            v_t = (self.v[i] / (1 - self.beta2 ** self.t))
            p.data = ndl.Tensor(p.numpy() - self.lr * m_t / (v_t ** 0.5 + self.eps), dtype="float32")
