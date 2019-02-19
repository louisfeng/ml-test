from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
mx.random.seed(1)
ctx = mx.cpu()

dim = 2
num_filter = 1
num_group = 1
kernel = (3,) * dim
print("kernel: ", kernel)
shape = (1, num_filter) + (3,) * dim
print("shape: ", shape)

x = mx.sym.Variable('x')
w = mx.sym.Variable('w')
b = mx.sym.Variable('b')
y1 = mx.sym.Convolution(data=x, weight=w, bias=b, num_filter=num_filter, num_group=num_group, kernel=kernel)
print(y1.list_arguments())

exe1 = y1.simple_bind(ctx, x=shape)
#x_data = mx.nd.ones(shape)
x_data = mx.nd.random.normal(shape=shape)
print("x: ", x_data)
#w_data = mx.nd.ones((1,num_filter)+kernel)
w_data = mx.nd.random.normal(shape=(1,num_filter)+kernel)
print("w: ", w_data)
#b_data = mx.nd.ones(num_filter)
b_data = mx.nd.random.normal(shape=num_filter)
print("b: ", b_data)
print(y1.debug_str())
fwd = exe1.forward(is_train=True, x = x_data, w=w_data, b=b_data)
print(fwd[0].asnumpy())

delta_data = exe1.outputs[0]
print("delta_data:", delta_data)
exe1.backward(delta_data)
print("backprop gradient")
print(exe1.grad_arrays[0].asnumpy())
print(exe1.grad_arrays[1].asnumpy())
print(exe1.grad_arrays[2].asnumpy())