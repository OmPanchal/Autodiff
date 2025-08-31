import numpy as np
from autodiff.nodes import Var, Const, Operator
from autodiff.Tensor import Tensor
from autodiff.ufunc_gradients import GRADS
import time
import pprint



class Graph(object):
	def __init__(self):
		self.variables = set()
		self.params = {}
		self.param_vals = {}
		self.__grad_funcs_string = {}
		self.grad_funcs = {}
		self.dy = None

	def __generate_grad_funcs(self): 
		for var, op in zip(self.__grad_funcs_string[self.__dy_grad_func_name].keys(), self.__grad_funcs_string[self.__dy_grad_func_name].values()):
			loc = {}
			func = f"def {var}({','.join(self.params)}, **kwargs):return {op}"
			try: 
				exec("import numpy as np")
				exec(func, None, loc)
				self.grad_funcs[self.__dy_grad_func_name][var] = loc[var]
			except SyntaxError:
				print("Expression too large to differentiate")
				exit()

	def add_variables(self, *args):
		for v in args:
			self.variables.add(v._source)
			self.param_vals[v._source.name] = v._i
			self.params[v._source.name] = v

	def __generate_grad_func_op(self, inp, grad):
		if inp.name in self.__grad_funcs_string[self.__dy_grad_func_name].keys():
				self.__grad_funcs_string[self.__dy_grad_func_name][inp.name] = \
					f"{self.__grad_funcs_string[self.__dy_grad_func_name][inp.name]}+{grad._source.string()}" 
		else:
				self.__grad_funcs_string[self.__dy_grad_func_name][inp.name] = grad._source.string()

	def __search(self, operation, variable, dtype="float32", dout=None):
		if dout is None:
			val = np.ones(shape=self.dy._i.shape, dtype=self.dy.dtype)
			dout = Tensor(val, source=Const(val))

		for idx, inp in enumerate(operation.inputs):
			if not GRADS.get(operation.optype):
				raise KeyError(f"The gradient for the operation: {operation.optype} has not been implemented yet")
			
			tensors = []
			for i in operation.inputs:
				tensors.append(Tensor(i.value, source=i, dtype=dtype))

			grad = GRADS[operation.optype][idx](*tensors, dout)

			if isinstance(inp, Operator):
				# if the source of dy is an operator...
				if inp.name in self.params.keys() and inp.name == variable.name:
					self.__generate_grad_func_op(inp, grad)
				self.__search(inp, variable, dtype, grad)

			elif isinstance(inp, Var):
				# if the variable appears more than once in the tree
				if inp.name == variable.name:
					self.__generate_grad_func_op(inp, grad)

	@property
	def __dy_grad_func_name(self):
		return f"{self.dy._source.name}_{str(self.n)}"

	def gradient(self, dy, dx, n=1):
		self.dy = dy
		self.n = n
		output = []

		for x in dx:
			prev_dy = self.dy

			try:
				func = self.grad_funcs[self.__dy_grad_func_name][x._source.name]
				output.append(func(**self.param_vals))

			except KeyError:
				for i in range(1, n + 1):
					self.n = i

					self.__grad_funcs_string[self.__dy_grad_func_name] = \
									self.__grad_funcs_string.get(self.__dy_grad_func_name) \
										if self.__grad_funcs_string.get(self.__dy_grad_func_name) != None else {}

					self.grad_funcs[self.__dy_grad_func_name] = \
									self.grad_funcs.get(self.__dy_grad_func_name) \
										if self.grad_funcs.get(self.__dy_grad_func_name) != None else {}

					if not self.grad_funcs[self.__dy_grad_func_name].get(x._source.name):
						try:
							self.__search(prev_dy._source, x._source, x.dtype)
							self.__generate_grad_funcs()
						except RecursionError:
							print("The maximum search depth has been reached. To increase it, increase the maximum recursion depth.")
							exit()
					
					try: 
						f = self.grad_funcs[self.__dy_grad_func_name][x._source.name]
						prev_dy = f(**self.params)
					except KeyError:
						# When the generate function has been generated, however no function has been created for the current variable.
						# This implies that the current variable was not part of dy's operations
						self.grad_funcs[self.__dy_grad_func_name][x._source.name] = lambda **kwargs: 0

				func = self.grad_funcs[self.__dy_grad_func_name][x._source.name]
				output.append(func(**self.param_vals))

		self.dy = None
		self.n = None
		self.reset_counts()
		return output
		
	def reset_counts(self):
		Var.reset_count()
		Const.reset_count()
		Operator.reset_count()
