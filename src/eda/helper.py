import numpy as np
import scipy.stats as stats
import pandas as pd

from data import repository as rep


class Helper(object):
	def __init__(self, train_name=None, test_name=None):
		super(Helper, self).__init__()
		self.train_name = 'train' if train_name is None else train_name
		self.test_name = 'test' if test_name is None else test_name

	def get_description(self, name, columns):
		df = rep.__getattr__(name).loc[:, columns]
		print(f"Shape: {df.shape}")
		description = pd.DataFrame(index=df.columns)
		description['Type'] = list(df.dtypes)
		description['Missing'] = df.isnull().sum().values
		description['Uniques'] = df.nunique().values
		description['Mean'] = None
		description['Std'] = None
		description['Min'] = None
		description['25%'] = None
		description['50%'] = None
		description['75%'] = None
		description['Max'] = None
		description['High freq'] = None
		description['Low freq'] = None
		description['Entropy'] = None

		for column in list(description.index):
			if str(description.loc[column, 'Type']) == 'category' or \
			   str(description.loc[column, 'Type']) == 'object':
				data = df[column].astype('object')
				description.loc[column, 'High freq'] = data.fillna('?').value_counts().sort_values().index[-1]
				description.loc[column, 'Low freq'] = data.fillna('?').value_counts().sort_values().index[0]
			else:
				data = df[column].astype('float32')
				description.loc[column, 'Mean'] = data.mean()
				description.loc[column, 'Std'] = data.std()
				description.loc[column, 'Min'] = data.min()
				description.loc[column, '25%'] = data.quantile(.25)
				description.loc[column, '50%'] = data.quantile(.5)
				description.loc[column, '75%'] = data.quantile(.75)
				description.loc[column, 'Max'] = data.max()
				description.loc[column, 'High freq'] = data.value_counts(dropna=False).sort_values().index[-1]
				description.loc[column, 'Low freq'] = data.value_counts(dropna=False).sort_values().index[0]

			description.loc[column, 'Entropy'] = stats.entropy(data.value_counts(dropna=False) / df.shape[0], base=2)

		return description

	def get_correlated_features(self, columns, val=.9):
		corr = []

		train = rep.__getattr__(self.train_name).loc[:, columns].corr()
		test = rep.__getattr__(self.test_name).loc[:, columns].corr()
		i = 0
		for col1 in columns:
			for col2 in columns[:i]:
				if train.loc[col1, col2] > val and test.loc[col1, col2] > val:
					corr.append((col1, col2,
					             train.loc[col1, col2], test.loc[col1, col2]))

			i += 1

		return pd.DataFrame(corr, columns=['x', 'y', 'c_train', 'c_test'])
