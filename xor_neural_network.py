import autodiff as ad
import numpy as np
import time


g = ad.Graph()

X = ad.Tensor([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]], dtype="float64")
Y = ad.Tensor([[0], [1], [1], [0]], dtype="float64")

W1 = ad.Tensor(np.random.rand(160, 2), dtype="float64")
B1 = ad.Tensor(np.random.rand(160, 1), dtype="float64")

W2 = ad.Tensor(np.random.rand(1, 160), dtype="float64")
B2 = ad.Tensor(np.random.rand(1, 1), dtype="float64")


def forward(x):
	Z1 = np.matmul(W1, x) + B1
	A1 = np.tanh(Z1)

	Z2 = np.matmul(W2, A1) + B2

	return Z2


_ = time.time()

for i in range(1000):
	for x, y in zip(X, Y):
		parameters = [W1, B1, W2, B2]

		g.add_variables(*parameters, x, y)

		Z2 = forward(x)

		E = (y - Z2) ** 2

		for param, gradient in zip(parameters, g.gradient(E, parameters)):
			param.assign(param._i - (0.01 * gradient))


print("training time:", time.time() - _)

print("predictions")
for x in X:
	print(forward(x))
