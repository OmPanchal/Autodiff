import numpy as np
from autodiff.nodes import Var, Const, Operator
from autodiff.Tensor import Tensor
from autodiff.ufunc_gradients import GRADS


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
		for var, op in zip(self.__grad_funcs_ops[self.dy._source.name].keys(), self.__grad_funcs_ops[self.dy._source.name].values()):
			loc = {}
			func = f"def {var}({','.join(self.params)}, **kwargs):return {op}"

			exec("import numpy as np")
			exec(func, None, loc)
			self.grad_funcs[self.dy._source.name][var] = loc[var]

	def add_variables(self, *args):
		for v in args:
			self.variables.add(v._source)
			self.params[v._source.name] = v

	def __generate_grad_func_op(self, inp, grad):
		if inp.name in self.__grad_funcs_ops[self.dy._source.name].keys():
			self.__grad_funcs_ops[self.dy._source.name][inp.name] = f"np.add({self.__grad_funcs_ops[self.dy._source.name][inp.name]}, {grad._source.string()})" 
		else:
			self.__grad_funcs_ops[self.dy._source.name][inp.name] = grad._source.string()

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
		self.__grad_funcs_ops[dy._source.name] = {}
		self.grad_funcs[dy._source.name] = {}
		output = []

		for x in dx:
			func_name = f"d{x._source.name}"

			if not self.grad_funcs.get(func_name):
				self.__search(dy._source)
				self.__generate_grad_funcs()

			func = self.grad_funcs[dy._source.name][x._source.name]
			output.append(func(**self.params))
		
		self.dy = None
		self.reset_counts()
		return output
		
	def reset_counts(self):
		Var.reset_count()
		Const.reset_count()
		Operator.reset_count()
