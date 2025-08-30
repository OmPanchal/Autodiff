import numpy as np
import autodiff as ad
import matplotlib.pyplot as plt
import time
import sys


# a = ad.Tensor(2)
a = ad.Tensor(np.arange(-2, 2, 0.01), dtype="float64")
b = ad.Tensor(3)
c = ad.Tensor(2)

# d = np.sin(np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a + np.tan(np.cos(np.cosh((a ** np.tan(np.cos(np.cosh((b ** 7) + (a ** 3) - a) + a ** 4) * a) - a) + (a ** 3) - a) + a ** 4) * a) - a)

d = np.sin(a) * np.cos(a)

# d = np.tanh(a ** 2)
 
g = ad.Graph()
g.add_variables(a, b, c)

# g.gradient(d, [a, b, c])
for i in range(5):
	plt.plot(g.gradient(d, [a], i + 1)[0])
plt.show()




# ! optimisation 1: if both the inputs of an operator are constants, then just calculate the output of the operation.
# ! optimisation 2: if any of a multiplication operation's inputs are 1 or an array of 1, then return only the other input. 

# sin(tan(cos(cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a + tan(cos(cosh((a ** tan(cos(cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a) + (a ** 3) - a) + a ** 4) * a) - a)