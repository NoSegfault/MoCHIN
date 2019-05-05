motifs = ['AP', 'PV', 'PT', 'PP']

lambdas = [1, 10, .001]

N_clusters = 5

input_prefix = 'input_group/'
seed_file = input_prefix + 'label-5-group.txt'

node_type_number = input_prefix + 'node-type-number.txt'
target = 'A'
target_list = input_prefix + 'single-author-list.txt'

labelfile_train = input_prefix + 'train_test_split_group/dblp-group-train-label.txt'
labelfile_test = input_prefix + 'train_test_split_group/dblp-group-test-label.txt'

motif_weights = [.25] * 4

# System Settings:
# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
loss_eval_step_size = 1

nprocesses_for_L1 = 1
nprocesses_for_U1 = 1