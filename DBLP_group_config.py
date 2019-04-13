class Config:
	motifs = ['AP', 'PV', 'PT', 'PP']
	seed_file = 'original_group_final/label-5-group.txt'
	N_clusters = 5
	nprocesses_for_L1 = 1
	nprocesses_for_U1 = 1
	list_prefix = 'intermediate_group_final/'
	single_author_list = 'original_group_final/single-author-list.txt'
	paper_list = list_prefix + 'paper-list-10.txt'
	venue_list = list_prefix + 'venue-list-10.txt'
	term_list = list_prefix + 'term-list-10.txt'
	numbers = list_prefix + 'numbers-12.txt'
	labelfile_train = 'train_test_split_group/dblp-group-train-label.txt'
	labelfile_test = 'train_test_split_group/dblp-group-test-label.txt'
