import mxnet as mx
x = mx.sym.Variable('x')
a = mx.sym.Variable('a')
y = x * a
j = y + x
z = y + j

e = z.bind(mx.cpu(), {'x': mx.nd.array([[1,2],[3,4]]), 'a':mx.nd.array([[1,2],[3,4]])}, args_grad={'x': mx.nd.empty((2, 2))})
e.forward(is_train=False)
e.backward(out_grads=[mx.nd.ones((2, 2))])
print(e.grad_arrays[0].asnumpy())
