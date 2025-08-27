import numpy as np
import autodiff as ad
import matplotlib.pyplot as plt


# a = ad.Tensor(1)
a = ad.Tensor(np.arange(-2.5, 2.5, 0.01), dtype="float64")
# b = Tensor(4)
# c = Tensor(2)

d = np.sin(np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a + np.tan(np.cos(np.cosh((a ** np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a) + (a ** 3) - a) + a ** 4) * a) - a)
# d = (a ** 3)
# d = np.tanh(a ** 2)
 
g = ad.Graph()
g.add_variables(a)

gradient1 = g.gradient(d, [a])[0]
print(gradient1)
gradient2 = g.gradient(gradient1, [a])[0]
print(gradient2)
# gradient3 = g.gradient(gradient2, [a])[0]
# gradient3 = g.gradient(gradient2, [a])[0]
# gradient3 = g.gradient(gradient2, [a])[0]
# gradient3 = g.gradient(gradient2, [a])[0]
# gradient3 = g.gradient(gradient2, [a])[0]
# gradient3 = g.gradient(gradient2, [a])[0]
# print(gradient3)

# print(g.grad_func_ops)
# plt.plot(d._i)
# plt.plot(gradient._i)

# plt.legend(["f(x)", "f'(x)"])
# plt.show()

# ! optimisation 1: if both the inputs of an operator are constants, then just calculate the output of the operation.
# ! optimisation 2: if any of a multiplication operation's inputs are 1 or an array of 1, then return only the other input. 