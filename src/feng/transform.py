import numpy as np
import pandas as pd
from data import repository as rep


class TransformerFunction(object):
	def __init__(self, func):
		super(TransformerFunction, self).__init__()
		self.func = func

	def __call__(self, *args, **kwargs):
		name = args[0]
		new_name = args[1]
		if len(args) == 2:
			replace = False if kwargs.get('replace') is None else kwargs['replace']
			use = [] if kwargs.get('use') is None else kwargs['use']
		elif len(args) < 4:
			replace = args[2]
			if len(args) == 3:
				use = [] if kwargs.get('use') is None else kwargs['use']
			else:
				use = args[3]
		else:
			print('Error')
			return

		if len(use) == 0:
			use = rep.names()

		for df_name in use:
			if name in rep.__getattr__(df_name).columns:
				df = rep.__getattr__(df_name)
				func_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in ['inplace', 'use']}
				df[new_name] = self.func(df, name, **func_kwargs)
				if replace:
					del df[name]


class FeatureTransformerBase(object):
	def __init__(self):
		super(FeatureTransformerBase, self).__init__()
	
	def __getattr__(self, name):
		if name.startswith('do_'):
			return TransformerFunction(self.__getattribute__(name[3:]))
		else:
			return None


class DtFeatureTransformer(FeatureTransformerBase):
	@staticmethod
	def weekday(df, name):
		return np.floor(df[name] / (3600 * 24) - 1) % 7

	@staticmethod
	def dayhour(df, name):
		return np.floor(df[name] / 3600) % 24


class NumFeatureTransformer(FeatureTransformerBase):
	@staticmethod
	def log(df, name):
		return df[name].apply(np.log)

	@staticmethod
	def decimal(df, name):
		return ((df[name] - df[name].astype(int)) * 1000).astype(int)

	@staticmethod
	def mean(df, name, mean_name):
		return df[name] / df.groupby([mean_name])[name].transform('mean')

	@staticmethod
	def std(df, name, std_name):
		return df[name] / df.groupby([std_name])[name].transform('std')


class StrFeatureTransformer(FeatureTransformerBase):
	@staticmethod
	def split(df, name, sep, ind, exp=True):
		return df[name].str.split(sep, expand=exp)[ind]

	@staticmethod
	def replace(df, name, pat, val, reg=False):
		return df[name].str.replace(pat, val, regex=reg)

	@staticmethod
	def concat(df, name, other_name, sym='_'):
		return df[name].astype(str) + sym + df[other_name].astype(str)
