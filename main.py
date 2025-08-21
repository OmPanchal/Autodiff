import numpy as np
from autodiff.Tensor import Tensor
from autodiff.nodes import Graph


a = Tensor(1, dtype="float64")
# b = Tensor(4)
# c = Tensor(2)

d = np.sin(np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a)

e = d * 2

g = Graph()

g.add_variables(a, d)

# gradient = g.gradient(d, [a])
gradient = g.gradient(e, [a])
print(gradient)