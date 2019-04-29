class Config:
	motifs = ['P1WW1P', 'P1OL', 'P3OL', 'P2OL', 'P1P2P3W', 'P1O', 'P2O', 'P3O', 'O1L', 'P1P', 'P3P', 'P4P', 'P5P', 'P1W', 'P2W', 'P3W', 'P4W', 'P5W', 'P1R', 'P1S']
	
	motif_weights = [.05] * 20
	
	lambdas = [1, 10, .001]

	N_clusters = 10

	original_prefix = 'original_yago/'
	seed_file = original_prefix + 'yago_label_country.txt'
	list_prefix = 'intermediate_yago/'

# 	node_type_number = {'P': N_persons, 'W': N_works, 'O': N_orgs, 'R': N_prizes, 'S': N_positions, 'L': N_locs}
	node_type_number = {'P': 11368, 'W': 1794, 'O': 3711, 'R': 238, 'S': 11, 'L': 10}
	target = 'P'
	target_list = list_prefix + 'person_l.txt'
	
	labelfile_train = 'train_test_yago/yago-area-train-label.txt'
	labelfile_test = 'train_test_yago/yago-area-test-label.txt'
	
	# System Settings:
	# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
	loss_eval_step_size = 1
	
	nprocesses_for_L1 = 1
	nprocesses_for_U1 = 1
