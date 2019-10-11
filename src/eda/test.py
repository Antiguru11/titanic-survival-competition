from eda import helper, plotter
from data import repository as rep

import scipy.stats as stats

if __name__ == '__main__':
	rep.use(['train', 'test'])
	rep.optimize()
	rep.bootstrap()

	corr = helper.get_correlated_features(['C1', 'C2'])
	print('Done!')
