from feng import *
from data import repository as rep

if __name__ == '__main__':
	rep.use(names=['train'])
	rep.bootstrap()

	# dt_trans = DtFeatureTransformer()
	# dt_trans.make_TransactionDT_weekday(use=['train'])
	# print(rep.train[['TransactionDT', 'TransactionDT_weekday']].iloc[:5])

	numt = NumFeatureTransformer()
	numt.do_decimal('TransactionAmt', 'TransactionAmt_decimal')
	print(rep.train[['TransactionAmt', 'TransactionAmt_decimal']].iloc[:5])

	# num_trans.make_TransactionAmt_log(use=['train'])
	# print(rep.train[['TransactionAmt', 'TransactionAmt_log']].iloc[:5])
	#
	# num_trans.make_TransactionAmt_mean_card1(use=['train'], mean_name='card1')
	# print(rep.train[['TransactionAmt', 'TransactionAmt_by_mean_card1']].iloc[:5])
	#
	# num_trans.make_TransactionAmt_std_card1(use=['train'], std_name='card1')
	# print(rep.train[['TransactionAmt', 'TransactionAmt_by_std_card1']].iloc[:5])

	# str_trans = StrFeatureTransformer()
	# str_trans.do_split('DeviceInfo', 'device_name', sep='/', ind=0)
	# print(rep.train[['DeviceInfo', 'device_name']].iloc[:5])
	#
	# str_trans.do_split('DeviceInfo', 'device_version', sep='/', ind=1)
	# print(rep.train[['DeviceInfo', 'device_version']].iloc[:5])
	#
	# str_trans.do_replace('device_name', 'device_name', pat=r'(SAMSUNG|SM)(\S|\s)*', val='Samsung', reg=True)
	# print(rep.train[['DeviceInfo', 'device_name']].iloc[:5])
	fe_save()
