import numpy as np
import pprint
import time
import matplotlib.pyplot as plt
from numbers import Number

def absolute_grad(a, dout):
	return np.multiply(dout, (a / np.abs(a)))

def add_grad(a, b, dout):
	return dout

def arccos_grad(a, dout):
	return np.multiply(dout, -(1 / (np.sqrt(1 - (a ** 2)))))

def arccosh_grad(a, dout):	
	return np.multiply(dout, (1 / (np.sqrt((a ** 2) - 1))))

def arcsin_grad(a, dout):
	return np.multiply(dout, (1 / (np.sqrt(1 - (a ** 2)))))

def arcsinh_grad(a, dout):
	return np.multiply(dout, (1 / (np.sqrt((a ** 2) + 1))))

def arctan_grad(a, dout):
	return np.multiply(dout, (1 / ((a ** 2) + 1)))

def arctanh_grad(a, dout):
	return np.multiply(dout, (1 / (1 - (a ** 2))))

def cbrt_grad(a, dout):
	return np.multiply(dout, (1 / ((3 * a) ** (2 / 3))))

def cos_grad(a, dout):
	return np.multiply(dout, -np.sin(a))

def cosh_grad(a, dout):
	return np.multiply(dout, np.sinh(a))

def divide_grad_left(a, b, dout):
	return np.divide(dout, b)

def divide_grad_right(a, b, dout):
	return -np.multiply(dout, (a / (b ** 2)))

def exp2_grad(a, dout):
	return np.multiply(dout, (np.log(2) * (2 ** a)))

def exp_grad(a, dout):
	return np.multiply(dout, np.exp(a))

def log10_grad(a, dout):
	return np.multiply(dout, (1 / (np.log(10) * a)))

def log1p_grad(a, dout):
	return np.multiply(dout, (1 / (a + 1)))

def log2_grad(a, dout):
	return np.multiply(dout, (1 / np.log(2) * a))

def log_grad(a, dout):
	return np.multiply(dout, (1 / a))

# def matmul_grad(a, b, dout):

def multiply_grad_left(a, b, dout):
	return np.multiply(dout, b)

def multiply_grad_right(a, b, dout):
	return np.multiply(dout, a)

def negative_grad(a, dout):
	return np.negative(dout)

def power_grad_left(a, b, dout):
	return np.multiply(dout, (b * (a ** (b - 1))))

def power_grad_right(a, b, dout):
	return np.nan_to_num(np.multiply(dout, (a ** b) * (np.log(a))), nan=0, neginf=0, posinf=0)

def reciprocal_grad(a, dout):
	return np.multiply(dout, -np.reciprocal(a ** 2))

def sin_grad(a, dout):
	return np.multiply(dout, np.cos(a))

def sinh_grad(a, dout):
	return np.multiply(dout, np.cosh(a))

def sqrt_grad(a, dout):
	return np.multiply(dout, 1 / (2 * np.sqrt(a)))

def subtract_grad_left(a, b, dout):
	return dout

def subtract_grad_right(a, b, dout):
	return -dout

def tan_grad(a, dout): 
	return np.multiply(dout, 1 + (np.tan(a) ** 2))

def tanh_grad(a, dout):
	return np.multiply(dout, 1 - (np.tanh(a) ** 2))


def nan_to_num(x, copy=True, nan=0, posinf=None, neginf=None):
	return Tensor(np.nan_to_num(x._i, copy, nan, posinf, neginf), dtype=x.dtype, source=x._source)

HANDLED_FUNCTIONS = {
	np.nan_to_num: nan_to_num
}


GRADS = {
	np.abs.__name__: [absolute_grad],
	np.absolute.__name__: [absolute_grad],
	np.add.__name__: [add_grad, add_grad],
	np.arccos.__name__: [arccos_grad],
	np.arccosh.__name__: [arccosh_grad],
	np.arcsin.__name__: [arcsin_grad],
	np.arcsinh.__name__: [arcsinh_grad],
	np.arctan.__name__: [arctan_grad],
	np.arctanh.__name__: [arctanh_grad],
	np.cbrt.__name__: [cbrt_grad],
	np.cos.__name__: [cos_grad],
	np.cosh.__name__: [cosh_grad],
	np.divide.__name__: [divide_grad_left, divide_grad_right],
	np.exp2.__name__: [exp2_grad],
	np.exp.__name__: [exp_grad],
	np.log10.__name__: [log10_grad],
	np.log1p.__name__: [log1p_grad],
	np.log2.__name__: [log2_grad],
	np.log.__name__: [log_grad],
	np.multiply.__name__: [multiply_grad_left, multiply_grad_right],
	np.negative.__name__: [negative_grad],
	np.power.__name__: [power_grad_left, power_grad_right],
	np.reciprocal.__name__: [reciprocal_grad],
	np.sin.__name__: [sin_grad],
	np.sinh.__name__: [sinh_grad],
	np.sqrt.__name__: [sqrt_grad],
	np.subtract.__name__: [subtract_grad_left, subtract_grad_right],
	np.tan.__name__: [tan_grad],
	np.tanh.__name__: [tanh_grad],
}


class Graph(object):
	_g = None

	def __init__(self):
		self.nodes = set()


class Node(object):
	def __init__(self, value, name):
		self.value = value
		self._gradient = 0
		self.name = name

	def set_name(self, cls, name=None): return f"{name or cls.__name__[0]}{cls.count}"


class Operator(Node):
	count = 0

	def __init__(self, inputs, value=None, name=None, optype=None, *args, **kwargs):
		self.name = name or self.set_name(Operator, name)
		Operator.count += 1
		self.inputs = inputs
		self.inputs_strs = list(map(lambda x: x.string(), self.inputs))
		self.scalars = list(map(lambda x: x.value, self.inputs))
		self.optype = optype

		self._grad = GRADS.get(optype) or kwargs.get("grad")
		
		super().__init__(value, self.name)

	def __repr__(self):
		return f"<{Operator.__name__} name={self.name} type={self.optype}>"

	def gradient(self):
		return ...

	def string(self):
		return f"np.{self.optype}({','.join(self.inputs_strs)})"


class Var(Node):
	count = 0

	def __init__(self, value, name=None) -> None:
		self.name = name or self.set_name(Var, name) 
		Var.count += 1
		if Graph._g: Graph._g.add(self)
		super().__init__(value, self.name)
	
	def __repr__(self) -> str:
		return f"<Var name={self.name} value={self.value}>"
	
	def string(self):
		return self.name
	
	@staticmethod
	def reset_count():
		Var.count = 0


class Const(Node):
	count = 0

	def __init__(self, value, name=None):
		self.name = name or self.set_name(Const, name)
		Const.count += 1
		if Graph._g: Graph._g.add(self)
		super().__init__(value, name)

	def __repr__(self):
		return f"Const name={self.name} value={self.value}"
	
	def string(self):
		return f"{self.value}"


class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
	def __init__(self, _i, dtype="float32", *args, **kwargs):
		self._i = np.array(_i).astype(dtype)
		self.dtype = dtype
		self._source = kwargs.get("source") or Var(self._i, kwargs.get("name"))

	def __repr__(self):
		return f"<Tensor value={self._i} dtype={self.dtype}>"
	
	def __array__(self, dtype=None, copy=None):
		if copy is False:
			raise ValueError(
				"`copy=False` isn't supported. A copy is always created."
			)
		return np.array(self._i).astype(dtype)
	
	def __array_ufunc__(self, ufunc, method, *args, **kwargs):
		if method == '__call__':
			scalars, sources = self.__set_source_scalar(args)
			val = ufunc(*scalars, **kwargs)
			op = Operator(sources, value=val, name=None, optype=ufunc.__name__)
			return self.__class__(val, source=op)
		else:
			return NotImplemented
		
	def __array_function__(self, func, types, args, kwargs):
		if func not in HANDLED_FUNCTIONS:
			return NotImplemented

		if not all(issubclass(t, self.__class__) for t in types):
			return NotImplemented
			
		return HANDLED_FUNCTIONS[func](*args, **kwargs)

	def __set_source_scalar(self, inps):
		scalars = []
		sources = []

		for inp in inps:
			val = inp
			
			if issubclass(type(inp), Number):
				try: val = self.__class__(inp, source=Const(inp))
				except:
					raise ValueError(f"Cannot convert {inp} into type {self.__class__.__name__}")

			scalars.append(val._i)
			sources.append(val._source)

		return scalars, sources
	
	def copy(self):
		# shallow copy
		return self.__class__(self._i, dtype=self.dtype)



# print(np.arange(-10, 10, step=0.1))
a = Tensor(np.arange(-1, 1, step=0.00001))
# b = Tensor(4)
# c = Tensor(2)

d = np.sin(np.tan(np.cos(np.cosh((a ** 7) + (a ** 3) - a) + a ** 4) * a) - a)

op = [d._source]
paths = {}


class Graph(object):
	_g = None
	
	def __init__(self):
		self.nodes = set()
		self.variables = set()
		self.params = {}
		self.grad_funcs_ops = {}
		self.grad_funcs = {}
		self.dy = None
		Graph._g = self.nodes

	def generate_grad_funcs(self): 
		for var, op in zip(self.grad_funcs_ops.keys(), self.grad_funcs_ops.values()):
			loc = {}
			func_name = f"d{self.dy._source.name}_d{var}"
			func = f"def {func_name}({','.join(self.params)}, **kwargs):return {op}"

			exec("import numpy as np")
			exec(func, None, loc)
			self.grad_funcs[func_name] = loc[func_name]

	def add_variables(self, *args):
		for v in args:
			assert isinstance(v._source, Var) 
			self.variables.add(v._source)
			self.params[v._source.name] = v

	def search(self, operation, dout=Tensor(1, source=Const(1))):
		for idx, inp in enumerate(operation.inputs):
			if not GRADS.get(operation.optype):
				raise KeyError(f"The gradient for the operation: {operation.optype} has not been implemented yet")
			
			tensors = []
			for i in operation.inputs:
				tensors.append(Tensor(i.value, source=i))

			grad = GRADS[operation.optype][idx](*tensors, dout)

			if isinstance(inp, Operator):
				self.search(inp, grad)

			elif isinstance(inp, Var):
				if inp.name in self.grad_funcs_ops.keys():
					self.grad_funcs_ops[inp.name] = f"np.add({self.grad_funcs_ops[inp.name]}, {grad._source.string()})" 
				else:
					self.grad_funcs_ops[inp.name] = grad._source.string()

	def gradient(self, dy, dx):
		_ = time.time()
		self.dy = dy
		output = []

		for x in dx:
			func_name = f"d{dy._source.name}_d{x._source.name}"

			if not self.grad_funcs.get(func_name):
				self.search(dy._source)
				self.generate_grad_funcs()

			func = self.grad_funcs.get(func_name)
			output.append(func(**self.params))
		
		self.dy = None
		print("Took: ", time.time() - _)
		return output
		


	
g = Graph()

g.add_variables(a)

gradient = g.gradient(d, [a])
print(gradient)


plt.plot(d._i)
plt.plot(gradient[0]._i)
plt.show()
