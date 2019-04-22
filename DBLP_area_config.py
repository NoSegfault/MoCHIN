class Config:
	motifs = ['AP', 'PV']

	# motif_weights can be None
	motif_weights = [.3, .7]

	# lambdas can be None
	lambdas = [1, 10, .001]

	original_prefix = 'original_area_final/'
	seed_file = original_prefix + 'label-14-area.txt'
	N_clusters = 14
	list_prefix = 'intermediate_area_final/'
	single_author_list = original_prefix + 'single-author-list.txt'
	paper_list = list_prefix + 'paper-list.txt'
	venue_list = list_prefix + 'venue-list.txt'
	numbers = list_prefix + 'numbers-4.txt'
	labelfile_train = 'train_test_split_area/dblp-area-train-label.txt'
	labelfile_test = 'train_test_split_area/dblp-area-test-label.txt'

	# System Settings:
	# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
	loss_eval_step_size = 1
	
	nprocesses_for_L1 = 1
	nprocesses_for_U1 = 1
