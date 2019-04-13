class Config:
	motifs = ['P1WW1P', 'P1OL', 'P3OL', 'P2OL', 'P1P2P3W', 'P1O', 'P2O', 'P3O', 'O1L', 'P1P', 'P3P', 'P4P', 'P5P', 'P1W', 'P2W', 'P3W', 'P4W', 'P5W', 'P1R', 'P1S']
	seed_file = 'original_yago/yago_label_country.txt'
	N_clusters = 10
	nprocesses_for_L1 = 1
	nprocesses_for_U1 = 1
	list_prefix = 'intermediate_yago/'
	numbers = list_prefix + 'numbers.txt'
	person_l = list_prefix + 'person_l.txt'
	loc_list = list_prefix + 'loc_list.txt'
	countries = ['AD:18735', 'AD:3402', 'AD:647', 'AD:738', 'AD:1652', 'AD:2673', 'AD:3983', 'AD:7110', 'AD:985', 'AD:701']
	labelfile_train = 'train_test_yago/yago-area-train-label.txt'
	labelfile_test = 'train_test_yago/yago-area-test-label.txt'
