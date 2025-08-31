import numpy as np
import autodiff as ad
import matplotlib.pyplot as plt


a = ad.Tensor(np.arange(-2, 2, 0.01), dtype="float64")

d = np.tanh(a)

g = ad.Graph()
g.add_variables(a)

labels = []

for i in range(5):
	plt.plot(g.gradient(d, [a], i + 1)[0])
	labels.append(f"derivative {i + 1}")

plt.legend(labels)
plt.show()
