class Config:
	motifs = ['AP', 'PV', 'PT', 'PP']

	motif_weights = [.25] * 4

	lambdas = [1, 10, .001]

	N_clusters = 5

	original_prefix = 'original_group_final/'
	seed_file = original_prefix + 'label-5-group.txt'
	list_prefix = 'intermediate_group_final/'

# 	node_type_number = {'P': N_papers, 'A': N_authors, 'V': N_venues, 'T': N_terms}
	node_type_number = {'P': 11138, 'A': 245, 'V': 1564, 'T': 6550}
	target = 'A'
	target_list = original_prefix + 'single-author-list.txt'

	labelfile_train = 'train_test_split_group/dblp-group-train-label.txt'
	labelfile_test = 'train_test_split_group/dblp-group-test-label.txt'

	# System Settings:
	# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
	loss_eval_step_size = 1
	
	nprocesses_for_L1 = 1
	nprocesses_for_U1 = 1
