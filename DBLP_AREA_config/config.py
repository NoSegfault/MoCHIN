motifs = ['AP', 'PV']

# set lambdas to None to use default values
lambdas = [1, 10, .001]

N_clusters = 14

input_prefix = 'input_area/'
seed_file = input_prefix + 'label-area.txt'

node_type_number = input_prefix + 'node-type-number.pklz'
target = 'A'
target_list = input_prefix + 'author-list.pklz'

labelfile_train = input_prefix + 'train_test_split_area/area-train-label.txt'
labelfile_test = input_prefix + 'train_test_split_area/area-test-label.txt'

# set motif_weights to None to use default values
motif_weights = [.3, .7]

# System Settings:
# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
loss_eval_step_size = 1

nprocesses_for_L1 = 1
nprocesses_for_U1 = 1
