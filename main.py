import numpy as np
from autodiff.Tensor import Tensor
from autodiff import Graph
import matplotlib.pyplot as plt

a = Tensor(np.arange(-2.5, 2.5, 0.01), dtype="float64")
# b = Tensor(4)
# c = Tensor(2)

# d = np.sin(np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a)
# d = (a ** 3)
d = np.tanh(a)

g = Graph()
g.add_variables(a)

gradient = g.gradient(d, [a])[0]

plt.plot(d._i)

for i in range(5):
	gradient = g.gradient(gradient, [a])[0]
	plt.plot(gradient._i)

plt.legend(["tanh", "first derivative", "second derivative", "third derivative", "fourth derivative", "fifth derivative"])
plt.show()