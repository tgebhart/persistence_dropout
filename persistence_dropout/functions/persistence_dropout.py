import torch
from torch.autograd.function import Function
import math





class StaticPersistenceDropout(Function):
    @staticmethod
    def forward(ctx, x, muls, train, l):
        mul = muls[l]
        ctx.save_for_backward(x, mul)
        if train:
            m = mul.numpy()
            # print('Percent Zero: {0:.2f}'.format(100*(1 - m.reshape((-1,m.size)).sum()/m.size)))
            # potentially only serve prediction along subgraph?
            if x.is_cuda:
                mul = mul.cuda()
            return x * mul
        else:
            return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mul = ctx.saved_tensors
        return grad_output.clone(), None, None, None


class InducedPersistenceDropout(Function):
    @staticmethod
    def forward(ctx, x, muls, train, l):
        mul = muls[l]
        ctx.save_for_backward(x, mul)
        if train:
            m = mul.numpy()
            # print('Percent Zero: {0:.2f}'.format(100*(1 - m.reshape((-1,m.size)).sum()/m.size)))
            # potentially only serve prediction along subgraph?
            if x.is_cuda:
                mul = mul.cuda()
            return x * mul
        else:
            return x

    @staticmethod
    def backward(ctx, grad_output):
        x, mul = ctx.saved_tensors
        return grad_output.clone(), None, None, None
