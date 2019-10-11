from data import repository as rep

if __name__ == '__main__':
	rep.use_latest_fe()

	# rep.use(['fe_test_20190908_123500', 'fe_train_20190908_123500'])
	# rep.alias('fe_test_20190908_123500', 'test')
	# rep.alias('fe_train_20190908_123500', 'train')
	# rep.optimize()
	# rep.bootstrap()

	for item in rep:
		print(item.shape)
		print(item.name_)
		print(item.path_)
