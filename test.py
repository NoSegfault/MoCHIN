import pickle
import gzip


if __name__ == '__main__':


	with gzip.open('model.pklz', 'rb') as f:
		ret = pickle.load(f)

	print(ret)
