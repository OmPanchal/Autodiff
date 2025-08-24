import numpy as np
from autodiff.Tensor import Tensor
from autodiff import Graph


a = Tensor(1, dtype="float64")
# b = Tensor(4)
# c = Tensor(2)

# d = np.sin(np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a)
d = (a ** 3)

g = Graph()

g.add_variables(a)

# gradient = g.gradient(d, [a])
gradient = g.gradient(d, [a])[0]
print(gradient)
print(g.grad_func_ops)
print(g.gradient(gradient, [a]))
print(g.grad_func_ops)
# print(g.grad_func_ops)