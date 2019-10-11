import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config as cfg
from data import repository as rep


class _Plotter(object):
	def __init__(self, train_name=None, test_name=None):
		super(_Plotter, self).__init__()
		self.fig_kwargs = {'figsize': (20, 10)}
		self.index = 1
		self.n_rows = 1
		self.n_cols = 1
		self.leg_kwargs = {}
		self.train_name = 'train' if train_name is None else train_name
		self.test_name = 'test' if test_name is None else test_name

	def get_column(self,
	               name,
	               column,
	               func=None,
	               flt=None,
	               dropna=True,
	               dtype=None):
		df = rep.__getattr__(name)
		if func is None:
			func = lambda x: x
		if flt is None:
			flt = lambda df: np.ones(len(df), dtype=bool)
		if dtype is None:
			dtype = df[column].dtype

		if dropna:
			df = df.dropna(subset=[column])

		return df.loc[flt, [column]].apply(func).astype(dtype)

	def set_figure(self, **kwargs):
		self.fig_kwargs = kwargs

	def set_grid(self, n_rows, n_cols):
		self.n_rows = n_rows
		self.n_cols = n_cols

	def set_legend(self, **kwargs):
		self.leg_kwargs = kwargs

	def target(self, **sns_kwargs):
		pass

	def dist(self, name, func=None, **sns_kwargs):
		self.index = 1
		plt.figure(**self.fig_kwargs)
		ax = plt.subplot(self.n_rows, self.n_cols, self.index)
		sns.distplot(self.get_column(self.train_name, name, func),
		             label='Train',
		             ax=ax,
		             **sns_kwargs)
		sns.distplot(self.get_column(self.test_name, name, func),
		             label='Test',
		             ax=ax,
		             **sns_kwargs)
		self.index += 1
		plt.legend(**self.leg_kwargs)

	def log_dist(self, name, **sns_kwargs):
		self.dist(name, lambda x: np.log(x), **sns_kwargs)

	def count(self, name, values=None, **sns_kwargs):
		flt = None if values is None else lambda df: df[name].isin(values)

		train, test = self.get_column('train', name, flt=flt, dtype='object'),\
		              self.get_column('test', name, flt=flt, dtype='object')
		train['Type'] = 'Train'
		test['Type'] = 'Test'
		total = pd.concat([train, test], sort=False)

		self.index = 1
		plt.figure(**self.fig_kwargs)
		ax = plt.subplot(self.n_rows, self.n_cols, self.index)
		if len(total) != 0:
			sns.countplot(x=name, hue='Type', ax=ax, data=total)
		self.index += 1
		plt.legend(**self.leg_kwargs)

	def corr(self, columns, **sns_kwargs):
		train, test = rep.__getattr__('train').loc[:, columns], \
		              rep.__getattr__('test').loc[:, columns]
		self.index = 1
		plt.figure(**self.fig_kwargs)
		sns.pairplot(train, **sns_kwargs)

		plt.figure(**self.fig_kwargs)
		sns.pairplot(test, **sns_kwargs)


class ClassificationPlotter(_Plotter):
	def __init__(self, train_name=None, test_name=None):
		super(ClassificationPlotter, self).__init__(train_name, test_name)

	def target(self, **sns_kwargs):
		train = self.get_column('train', cfg.target_col)

		plt.figure(**self.fig_kwargs)
		ax = sns.countplot(x=cfg.target_col, data=train, **sns_kwargs)
		total = len(train)
		for key, val in cfg.target_vars.items():
			cnt = len(train.loc[train[cfg.target_col] == val])
			ax.text(val,
			        cnt,
			        "{0} {1:.2f}%".format(key, cnt / total * 100),
			        fontsize=20,
			        ha='center')

		plt.show()

	def dist(self, name, func=lambda x: x, **sns_kwargs):
		self.set_grid(1, 2)
		super(ClassificationPlotter, self).dist(name, func, **sns_kwargs)

		ax = plt.subplot(self.n_rows, self.n_cols, self.index)
		for key, val in cfg.target_vars.items():
			sns.distplot(self.get_column(self.train_name,
			                             name,
			                             func,
			                             lambda df: df[cfg.target_col] == val),
			             label=f'Train - {key}',
			             ax=ax,
			             **sns_kwargs)
		self.index += 1
		plt.legend(**self.leg_kwargs)
		plt.show()

	def count(self, name, values=None, **sns_kwargs):
		self.set_grid(np.ceil(len(cfg.target_vars) / 2) + 1, 2)
		super(ClassificationPlotter, self).count(name, values, **sns_kwargs)
		self.index += 1

		for key, val in cfg.target_vars.items():
			flt = lambda df: df[cfg.target_col] == val
			if values is not None:
				flt = lambda df: (df[cfg.target_col] == val) & df[name].isin(values)
				
			data = self.get_column('train', name, flt=flt, dtype='object')
			ax = plt.subplot(self.n_rows, self.n_cols, self.index)
			if len(data) != 0:
				sns.countplot(x=name, label=key, ax=ax, data=data, **sns_kwargs)
			self.index += 1
			plt.legend(**self.leg_kwargs)
		plt.show()

	def corr(self, columns, **sns_kwargs):
		super(ClassificationPlotter, self).corr(columns, **sns_kwargs)

		plt.show()
