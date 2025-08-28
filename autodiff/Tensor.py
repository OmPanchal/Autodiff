import numpy as np
from numbers import Number
from autodiff.nodes import Operator, Var, Const


def nan_to_num(x, copy=True, nan=0, posinf=None, neginf=None):
	return Tensor(np.nan_to_num(x._i, copy, nan, posinf, neginf), dtype=x.dtype, source=x._source)

def transpose(a, **kwargs):
	val = np.transpose(a._i, **kwargs)
	op = Operator([a._source], value=val, name=None, optype=np.transpose.__name__)
	return Tensor(val, dtype=a.dtype, source=op)

HANDLED_FUNCTIONS = {
	np.nan_to_num: nan_to_num,
	np.transpose: transpose
}


class Tensor(np.lib.mixins.NDArrayOperatorsMixin):
	def __init__(self, _i, dtype="float32", *args, **kwargs):
		self._i = np.array(_i).astype(dtype)
		self.dtype = dtype
		self._source = kwargs.get("source") or Var(self._i, kwargs.get("name"))

	@property
	def T(self):
		return np.transpose(self)

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
			return self.__class__(val, dtype=self.dtype, source=op)
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
			
			if issubclass(type(inp), Number) or issubclass(type(inp), np.ndarray) or isinstance(inp, list):
				try: val = self.__class__(inp, source=Const(inp))
				except:
					raise ValueError(f"Cannot convert {inp} into type {self.__class__.__name__}")
			
			scalars.append(val._i)
			sources.append(val._source)

		return scalars, sources
	
	def copy(self):
		# shallow copy
		return self.__class__(self._i, dtype=self.dtype)
	
	def assign(self, a):
		self._i = a
