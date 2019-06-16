motifs = ['P1WW1P', 'P1OL', 'P3OL', 'P2OL', 'P1P2P3W', 'P1O', 'P2O', 'P3O', 'O1L', 'P1P', 'P3P', 'P4P', 'P5P', 'P1W', 'P2W', 'P3W', 'P4W', 'P5W', 'P1R', 'P1S']

lambdas = [1, 10, .001]

N_clusters = 10

input_prefix = 'input_yago/'
seed_file = input_prefix + 'label-yago.txt'

node_type_number = input_prefix + 'node-type-number.pklz' 
target = 'P'
target_list = input_prefix + 'person-list.pklz'

labelfile_train = input_prefix + 'train_test_split_yago/yago-train-label.txt'
labelfile_test = input_prefix + 'train_test_split_yago/yago-test-label.txt'

motif_weights = [.05] * 20

# System Settings:
# enable loss evaluation not at every inner-loop step (every k steps, where k is a variable)
loss_eval_step_size = 1

nprocesses_for_L1 = 1
nprocesses_for_U1 = 1
