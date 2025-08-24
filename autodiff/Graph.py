import numpy as np
from autodiff.nodes import Var, Const, Operator
from autodiff.Tensor import Tensor
from autodiff.ufunc_gradients import GRADS


# TODO: Need to change the way that the gradient functions are stored in the dictionary...
# currently the program only checks if a variable is present in the keys of the dictionary 
# which means that another derivative cannot be taken of it without its value being added to the previous derivative

class Graph(object):
	def __init__(self):
		self.variables = set()
		self.params = {}
		self.__grad_funcs_ops = {}
		self.grad_funcs = {}
		self.dy = None

	@property
	def grad_func_ops(self):
		return self.__grad_funcs_ops 

	def __generate_grad_funcs(self): 
		for var, op in zip(self.__grad_funcs_ops.keys(), self.__grad_funcs_ops.values()):
			loc = {}
			func_name = f"d{self.dy._source.name}_d{var}"
			func = f"def {func_name}({','.join(self.params)}, **kwargs):return {op}"

			exec("import numpy as np")
			exec(func, None, loc)
			self.grad_funcs[func_name] = loc[func_name]

	def add_variables(self, *args):
		for v in args:
			self.variables.add(v._source)
			self.params[v._source.name] = v

	def __generate_grad_func_op(self, inp, grad):
		if inp.name in self.__grad_funcs_ops.keys():
			self.__grad_funcs_ops[inp.name] = f"np.add({self.__grad_funcs_ops[inp.name]}, {grad._source.string()})" 
		else:
			self.__grad_funcs_ops[inp.name] = grad._source.string()

	def __search(self, operation, dout=None):
		if dout is None:
			val = np.ones(shape=self.dy._i.shape, dtype=self.dy.dtype)
			dout = Tensor(val, source=Const(val))

		for idx, inp in enumerate(operation.inputs):
			if not GRADS.get(operation.optype):
				raise KeyError(f"The gradient for the operation: {operation.optype} has not been implemented yet")
			
			tensors = []
			for i in operation.inputs:
				tensors.append(Tensor(i.value, source=i))

			grad = GRADS[operation.optype][idx](*tensors, dout)

			if isinstance(inp, Operator):
				# if the source of dy is an operator...
				if inp.name in self.params.keys():
					self.__generate_grad_func_op(inp, grad)
				self.__search(inp, grad)

			elif isinstance(inp, Var):
				# if the variable appears more than once in the tree
				self.__generate_grad_func_op(inp, grad)

	def gradient(self, dy, dx):
		self.dy = dy
		output = []

		for x in dx:
			func_name = f"d{dy._source.name}_d{x._source.name}"

			if not self.grad_funcs.get(func_name):
				self.__search(dy._source)
				# print(self.__grad_funcs_ops)
				self.__generate_grad_funcs()

			print(self.grad_funcs)

			func = self.grad_funcs.get(func_name)
			output.append(func(**self.params))
		
		self.dy = None
		self.reset_counts()
		return output
		
	def reset_counts(self):
		Var.reset_count()
		Const.reset_count()
		Operator.reset_count()
