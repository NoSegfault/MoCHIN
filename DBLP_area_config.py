class Config:
	motifs = ['AP', 'PV']

	# set motif_weights to None to use default values
	motif_weights = [.3, .7]

	# set lambdas to None to use default values
	lambdas = [1, 10, .001]

	N_clusters = 14

	original_prefix = 'original_area_final/'
	seed_file = original_prefix + 'label-14-area.txt'
	list_prefix = 'intermediate_area_final/'

# 	node_type_number = {'P': N_papers, 'A': N_authors, 'V': N_venues, 'T': N_terms}
	node_type_number = {'P': 2790, 'A': 7165, 'V': 36, 'T': 6109}
	target = 'A'
	target_list = original_prefix + 'single-author-list.txt'
	
	labelfile_train = 'train_test_split_area/dblp-area-train-label.txt'
	labelfile_test = 'train_test_split_area/dblp-area-test-label.txt'

	# System Settings:
	# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
	loss_eval_step_size = 1
	
	nprocesses_for_L1 = 1
	nprocesses_for_U1 = 1
