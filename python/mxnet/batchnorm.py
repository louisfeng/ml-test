from __future__ import print_function
import mxnet as mx
import numpy as np
from mxnet import nd, autograd
mx.random.seed(1)
ctx = mx.cpu()

def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    print("gamma", gamma)
    print("beta", beta)
    # dense
    if len(X.shape) == 2:
        C, N = X.shape
        # mini-batch mean
        # mini-batch mean
        mean = nd.mean(X, axis=0)
        print("mean:", mean)
        # mini-batch variance
        variance = nd.mean((X - mean) ** 2, axis=0)
        print("var:", variance)
        # normalize
        X_hat = (X - mean) * 1.0 / nd.sqrt(variance + eps)
        # scale and shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        # mini-batch mean
        mean = nd.mean(X, axis=(0, 2, 3))
        print("mean", mean)
        # mini-batch variance
        variance = nd.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        print("variance", variance)
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / nd.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        #X_hat = (X - mean.reshape((1, C, 1, 1)))
        print ("X_hat", X_hat)
        #print(X_hat)
        # scale and shift
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))
    return out
shape = [3,2]
axis = 1
data = nd.array([-3,-2,-1,1,2,3], ctx=ctx).reshape(shape)
print(data.asnumpy())
gamma = mx.nd.ones(2, ctx=ctx)
beta = mx.nd.zeros(2, ctx=ctx)
mm = mx.nd.ones(2, ctx=ctx)
mv = mx.nd.ones(2, ctx=ctx)
eps = 0.001
B = pure_batch_norm(data,
    gamma=gamma,
    beta=beta,
    eps=eps)
print(B)
print("NGRAPH Op")
print(gamma)
print(beta)
C = mx.sym.BatchNorm(name="bn1", axis=axis, fix_gamma=False, use_global_stats=False, momentum=0.9, eps=eps)
#print(C.list_arguments())
print("outputs:")
C.get_internals().list_outputs()
executor = C.bind(ctx=ctx, args={"bn1_data":data, "bn1_gamma":gamma, "bn1_beta":beta}, aux_states={"bn1_moving_mean":mm, "bn1_moving_var":mv})
#print(C.debug_str())
R = executor.forward(is_train=True)
print(R[0].asnumpy())
R = executor.forward(is_train=True)
print(R[0].asnumpy())
#C = mx.nd.BatchNorm(name="bn1", axis=1, data=data, gamma=gamma, beta=beta, moving_mean=mm, moving_var=mv, fix_gamma=True, use_global_stats=False, momentum=0.9, eps=eps)
#print("ndarray batchnorm:\n", C)
#
n = 2
c = 3
h = 4
w = 5
gamma = mx.nd.ones(c, ctx=ctx)*0.5
beta = mx.nd.zeros(c, ctx=ctx)
mm = mx.nd.zeros(c, ctx=ctx)
mv = mx.nd.zeros(c, ctx=ctx)
dataSize = n * c * h * w
data = nd.arange(dataSize).reshape([n, c, h, w])
#print(data.asnumpy())
B = pure_batch_norm(data, gamma=gamma, beta=beta)
print("pure_batch_norm\n", B.asnumpy())


axis = 1

C = mx.sym.BatchNorm(name="bn1", axis=axis, fix_gamma=0, use_global_stats=False, momentum=0.9, eps=eps)
print("NGRAPH Op")
executor = C.bind(ctx=ctx, args={"bn1_data":data, "bn1_gamma":gamma, "bn1_beta":beta}, aux_states={"bn1_moving_mean":mm, "bn1_moving_var":mv})
R = executor.forward(is_train=False)
print("ngraph batchnorm\n",R[0].asnumpy())
