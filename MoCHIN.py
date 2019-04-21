import numpy as np
import GradDescent as gd
import pickle
import time
import itertools
from joblib import Parallel, delayed
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import log_loss
import gzip
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("flag")
args = parser.parse_args()

DEBUG = True

if args.task == 'DBLP_GROUP':
	from DBLP_group_config import *
	random.seed(0)
elif args.task == 'DBLP_AREA':
	from DBLP_area_config import *
	random.seed(0)
elif args.task == 'YAGO':
	from YAGO_config import *
	random.seed(15)

seed_file = Config.seed_file
N_clusters = Config.N_clusters
labelfile_train = Config.labelfile_train
labelfile_test = Config.labelfile_test
[lambda1, lambda2, lambda3] = Config.lambdas
motifs = Config.motifs
motif_weights = Config.motif_weights
assert(len(motifs) == len(motif_weights))

loss_eval_step_size = Config.loss_eval_step_size
nprocesses_for_U1 = Config.nprocesses_for_U1
nprocesses_for_L1 = Config.nprocesses_for_L1

if DEBUG:
	print('[lambda1, lambda2, lambda3] is: {}'.format([lambda1, lambda2, lambda3]))
	print("Motifs: {}".format(motifs))
	print('Motif Init Weights: {}'.format(motif_weights))


N_motif_type = len(motifs)

all_indices_list = []

def number_remove(m):
	m_new = ''
	for c in m:
		if(c.isalpha()):
			m_new += c

	return m_new

###########################################
#Xinwei edited

def motif_load(m, motif_idx):

	print('motif_load\tmotif type:{}'.format(motif_idx))

	ret = []
	with gzip.open(Config.list_prefix + 'indices-list-{}.pklz'.format(m), 'rb') as f:
		ret = pickle.load(f)
	return ret


def label_author(labelfile):
	with open(labelfile, "r") as f:
		labels = {}
		for line in f:
			kv = line.split("\t")
			kv[0] = kv[0].lower()
			kv[1] = int(kv[1][:-1])
			labels[kv[0]] = kv[1]

	#f.close()
	return labels

def label_author_new(labelfile_train, labelfile_test):
	labels = {}
	seed_label = {}
	labels_dict_test = {}
	with open(labelfile_train, "r") as f:
		for line in f:
			kv = line.split("\t")
			if args.task != 'YAGO':
				kv[0] = kv[0].lower()
			kv[1] = int(kv[1][:-1])
			seed_label[kv[0]] = kv[1]


	with open(labelfile_test, "r") as f:
		for line in f:
			kv = line.split("\t")
			if args.task != 'YAGO':
				kv[0] = kv[0].lower()
			kv[1] = int(kv[1][:-1])
			labels_dict_test[kv[0]] = kv[1]

	labels = {**seed_label, **labels_dict_test}

	return labels, seed_label, labels_dict_test

###########################################
with open(Config.numbers, 'rb') as nf:
	numbers = pickle.load(nf)

if args.task[:4] == 'DBLP':
	[N_papers, N_authors, N_venues, N_terms] = numbers
	node_type_number = {'P': N_papers, 'A': N_authors, 'V': N_venues, 'T': N_terms}
elif args.task == 'YAGO':
	[N_persons, N_works, N_orgs, N_prizes, N_positions, N_locs] = numbers
	node_type_number = {'P': N_persons, 'W': N_works, 'O': N_orgs, 'R': N_prizes, 'S': N_positions, 'L': N_locs}


#########################################

# type_dictionary is T in the paper
type_dictionary = {}
for i in range(N_motif_type):
	type_dictionary[i] = {}
	m = motifs[i]
	for j in range(len(m)):
		type_dictionary[i][j] = motifs[i][j]

###########################################
def init_V_DBLP(seed_label, single_author_list):
	V = []

	V_author = np.zeros((node_type_number['A'], N_clusters))

	for i in range(N_authors):
		a = single_author_list[i]
		if a in seed_label:
			label = seed_label[a]
			V_author[i][:] = 0
			V_author[i][label-1] = 1
		else:
			V_author[i] = np.ones(N_clusters) * (1 / N_clusters)**0.5


	for i in range(N_motif_type):
		V.append([])
		m = motifs[i]
		# print(m)
		for j in range(len(m)):
			if m[j] == 'A':
				V_jm = V_author.copy()
			else:
				N_rows = node_type_number[m[j]]
				V_jm = np.full((N_rows, N_clusters), (1 / N_clusters) ** 0.5)
			
			V[i].append(V_jm)

	return V

def init_V_YAGO(seed_label, single_author_list, seed_label_cnt, labels_dict_test_cnt, loc_list):
	#get the maximum value from seed_label_cnt
	seed_label_cnt_max = 0
	seed_label_cnt_min = 1000000000
	for key in seed_label_cnt:
		if(seed_label_cnt_max < seed_label_cnt[key]):
			seed_label_cnt_max = seed_label_cnt[key]

		if(seed_label_cnt_min > seed_label_cnt[key]):
			seed_label_cnt_min = seed_label_cnt[key]

	#print('seed_label_cnt_max is: {}'.format(seed_label_cnt_max))
	#print('seed_label_cnt_min is: {}'.format(seed_label_cnt_min))

	seed_label_cnt_var = {}
	#amplifier = 1000
	amplifier = 10
	multiplier = 0.5
	for key in seed_label_cnt:
		#v0
		seed_label_cnt_var[key] = amplifier

	print('seed_label_cnt_var is: {}'.format(seed_label_cnt_var))

	labels_dict_test_cnt_sum = 0
	default_dist = np.ones(N_clusters)

	V = []

	V_author = np.zeros((node_type_number['P'], N_clusters))

	for i in range(N_persons):
		a = single_author_list[i]
		if a in seed_label:
			label = seed_label[a]
			V_author[i][:] = 0
			#V_author[i][label-1] = 1
			V_author[i][label-1] = seed_label_cnt_var[label]
		else:
			V_author[i] = np.ones(N_clusters) * (amplifier / N_clusters)**multiplier
			#V_author[i] = np.ones(N_clusters) * (amplifier / N_clusters)
			#V_author[i] = default_dist

	V_location = np.zeros((node_type_number['L'], N_clusters))
	for i in range(N_locs):
		V_location[i][:] = 0
		V_location[i][addr_transform[loc_list[i]]-1] = amplifier

	for i in range(N_motif_type):
		V.append([])
		m = motifs[i]
		#Xinwei edited
		# print(m)
		m_new = number_remove(m)
		#for j in range(len(m)):
		for j in range(len(m_new)):
			#print(m[j])
			#if(not m[j].isalpha()):
			#       continue
			#if m[j] >= '1' and m[j] <= '6':
			#       continue

			#if m[j] == 'P':
			if m_new[j] == 'P':
				V_jm = V_author.copy()
			elif m_new[j] == 'L':
				V_jm = V_location.copy()
			else:
				#N_rows = node_type_number[m[j]]
				N_rows = node_type_number[m_new[j]]
				V_jm = np.full((N_rows, N_clusters), (amplifier / N_clusters) ** multiplier)
				#V_jm = np.full((N_rows, N_clusters), (amplifier / N_clusters))


			V[i].append(V_jm)

	return V


def init_miu():
	# miu = np.random.dirichlet(np.ones(N_motif_type),size=1)
	# miu = np.ones(N_motif_type) / N_motif_type
	miu = np.array(motif_weights)
	return miu
############################################

'''
e.g. type_count_in_motifs["AAPP"] = {'A':2, 'P':2}
	 type_count_in_motifs["APPPV"] = {'A':1, 'P':3, 'V':1}
'''
type_count_in_motifs = {}
for i in range(N_motif_type):
	m = motifs[i]
	type_count_in_motifs[m] = {}
	if args.task == 'YAGO':
		m_new = number_remove(m)
		for j in range(len(m_new)):
			node_type = m_new[j]
			if node_type not in type_count_in_motifs[m]:
				type_count_in_motifs[m][node_type] = 1
			else:
				type_count_in_motifs[m][node_type] += 1
	elif args.task[:4] == 'DBLP':
		for j in range(len(m)):
			node_type = type_dictionary[i][j]
			if node_type not in type_count_in_motifs[m]:
				type_count_in_motifs[m][node_type] = 1
			else:
				type_count_in_motifs[m][node_type] += 1

#############################################

# M^T is the mask for node type T
def get_M(author_list, seed_label):
	M = {}
	for node_type in node_type_number:
		mask_mtr = np.zeros((node_type_number[node_type], N_clusters))
		M[node_type] = mask_mtr
		if node_type == 'A':
			count = 0
			for a in author_list:
				if a in seed_label:

					idx = author_list.index(a)
					count += 1
					lb = seed_label[a]-1
					for i in list(range(lb)) + list(range(lb+1, N_clusters)):
						mask_mtr[idx][i] = 1

			# print("{} labels for Mask".format(count))

	return M


#########################################################################

def get_V_two_stars(V, miu, motif_idx, mtr_idx, V_star):
	m = motifs[motif_idx]
	m_new = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
	node_type = m_new[mtr_idx]
	V_two_stars = V_star[node_type] - miu[motif_idx] / type_count_in_motifs[m][node_type] * V[motif_idx][mtr_idx]
	return V_two_stars


def get_V_star(V, miu):
	V_star = {}
	for node_type in node_type_number:
		V_star[node_type] = 0
		for i in range(N_motif_type):
			m = motifs[i]
			m_new = m
			if args.task == 'YAGO':
				m_new = number_remove(m)
			for j in range(len(m_new)):
				if m_new[j] == node_type:
					V_star[node_type] += miu[i] * V[i][j] / type_count_in_motifs[m][node_type]
	return V_star


#########################################################################
def multi_getL1(V, m_indices_list_part, motif_idx):
	# ts = time.time()
	m = motifs[motif_idx]
	acu = 0
	for indices in m_indices_list_part:
		vec = np.ones(N_clusters)
		if args.task[:4] == 'DBLP':
			for i in range(len(m)):
				vec *= V[motif_idx][i][indices[i]]
		elif args.task == 'YAGO':
			m_new = number_remove(m)
			for i in range(len(m_new)):
				vec *= V[motif_idx][i][indices[i]]
		acu += 2*np.sum(vec)
	# te = time.time()
	return acu


def getL1_sp(V, motif_idx, m_indices_list):
	m = motifs[motif_idx]
	motif_size = len(m)
	if args.task == 'YAGO':
		m_new = number_remove(m)
		motif_size = len(m_new)
	num_total_indices = len(m_indices_list)
	
	L1 = num_total_indices


	t3 = time.time()

	indices_per_process = int(num_total_indices / nprocesses_for_L1)
	
	acu = Parallel(n_jobs=nprocesses_for_L1)\
			(delayed(multi_getL1)
				(V, m_indices_list[indices_per_process*i : indices_per_process*(i+1) if i < nprocesses_for_L1-1 else num_total_indices], motif_idx)
				for i in range(nprocesses_for_L1))

	part2_sum = sum(acu)
	
	t4 = time.time()
	print("multiprocess get {} in {} seconds".format(part2_sum, t4-t3))
	L1 -= part2_sum
	
	part3_sum = 0
	for i in range(N_clusters):
		for j in range(N_clusters):
			prod = 1
			for k in range(motif_size):
				first_vec = V[motif_idx][k][:,i]
				second_vec = V[motif_idx][k][:,j]
				prod *= np.linalg.norm(np.multiply(first_vec**0.5, second_vec**0.5))**2
			part3_sum += prod
	L1 += part3_sum
	t5 = time.time()
	return L1



def getL2(V, V_star, motif_idx):
	L2 = 0
	m = motifs[motif_idx]
	mm = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
		mm = m_new
	for i in range(len(mm)):
		node_type = mm[i]
		L2 += np.linalg.norm(V[motif_idx][i] - V_star[node_type], ord='fro')**2
	return lambda1 * L2


def getL3(M, V_star):
	L3 = 0
	for node_type in node_type_number:
		L3 += np.linalg.norm(np.multiply(M[node_type], V_star[node_type]), ord='fro')**2
	return lambda2 * L3

def getL4(V, motif_idx):
	L4 = 0
	m = motifs[motif_idx]
	mm = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
		mm = m_new
	for i in range(len(mm)):
		L4 += np.linalg.norm(V[motif_idx][i], ord=1)
	return lambda3 * L4


#########################################################################
def prepare_T_vecs_DBLP():
	dict_list = []
	dict_range_processes = []

	for motif_idx in range(N_motif_type):
		#Xinwei edited
		m = motifs[motif_idx]

		m_indices_list = motif_load(m, motif_idx)
		all_indices_list.append(m_indices_list)

		print("{} has {} instances".format(m, len(m_indices_list)))

		dict_list.append([])
		dict_range_processes.append([])

		for mtr_idx in range(len(m)):
			indices_wo_j_to_j = {}
			for indices in m_indices_list:
				key = tuple([indices[i] for i in [x for j, x in enumerate(range(len(m))) if j != mtr_idx]])
				if key not in indices_wo_j_to_j:
					indices_wo_j_to_j[key] = [indices[mtr_idx]]
				else:
					indices_wo_j_to_j[key].append(indices[mtr_idx])

			dict_len = len(indices_wo_j_to_j)
			dict_len_per_process = int(dict_len / nprocesses_for_U1)

			dict_list[motif_idx].append(indices_wo_j_to_j)

			part_dict = []
			for i in range(nprocesses_for_U1):
				(start, end) = (i * dict_len_per_process, (i+1) * dict_len_per_process if i < nprocesses_for_U1-1 else dict_len)
				part_dict.append({mk: indices_wo_j_to_j[mk] for mk in (list(indices_wo_j_to_j))[start : end]})
			dict_range_processes[motif_idx].append(part_dict)

	return (dict_list, dict_range_processes)

def prepare_T_vecs_YAGO(labels_dict, all_person_l):
	dict_list = []
	dict_range_processes = []

	for motif_idx in range(N_motif_type):
		m = motifs[motif_idx]

		m_indices_list = motif_load(m, motif_idx)
		all_indices_list.append(m_indices_list)

		print("{} has {} instances".format(m, len(m_indices_list)))


		#Xinwei edited
		#m = motifs[motif_idx]
		#print('m is: {}'.format(m))
		m = number_remove(m)

		dict_list.append([])
		dict_range_processes.append([])

		for mtr_idx in range(len(m)):
			indices_wo_j_to_j = {}

			for indices in m_indices_list:
				key = tuple([indices[i] for i in [x for j, x in enumerate(range(len(m))) if j != mtr_idx]])
				if key not in indices_wo_j_to_j:
					indices_wo_j_to_j[key] = [indices[mtr_idx]]
				else:
					indices_wo_j_to_j[key].append(indices[mtr_idx])

			dict_len = len(indices_wo_j_to_j)
			dict_len_per_process = int(dict_len / nprocesses_for_U1)

			dict_list[motif_idx].append(indices_wo_j_to_j)

			part_dict = []
			for i in range(nprocesses_for_U1):
				(start, end) = (i * dict_len_per_process, (i+1) * dict_len_per_process if i < nprocesses_for_U1-1 else dict_len)
				part_dict.append({mk: indices_wo_j_to_j[mk] for mk in (list(indices_wo_j_to_j))[start : end]})
			dict_range_processes[motif_idx].append(part_dict)


	#exit(0)
	return (dict_list, dict_range_processes)

def multi_getU1D1(V, motif_idx, mtr_idx, part_indices_wo_j_to_j):
	ts = time.time()
	m = motifs[motif_idx]
	mm = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
		mm = m_new
	node_type = mm[mtr_idx]
	part_U1 = np.zeros((node_type_number[node_type], N_clusters))

	for key in part_indices_wo_j_to_j:
		G_star_vec = np.ones(N_clusters)

		for ite, j in enumerate([x for k, x in enumerate(range(len(mm))) if k != mtr_idx]):
			G_star_vec = np.multiply(G_star_vec, V[motif_idx][j][key[ite]])

		for i in part_indices_wo_j_to_j[key]:
			part_U1[i] += G_star_vec

	te = time.time()
	return part_U1


def get_U1_D1(V, motif_idx, mtr_idx, indices_wo_j_to_j, dict_range_processes):

	m = motifs[motif_idx]
	m_new = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
	shape = V[motif_idx][mtr_idx]
	node_type = m_new[mtr_idx]
	t1 = time.time()
	U1 = np.zeros((node_type_number[node_type], N_clusters))
	part_U1 = Parallel(n_jobs=nprocesses_for_U1)\
				(delayed(multi_getU1D1)
					(V, motif_idx, mtr_idx, dict_range_processes[i])
					for i in range(nprocesses_for_U1))

	U1 = sum((part_U1))

	t3 = time.time()
	print("{}-processes takes {} seconds for U1".format(nprocesses_for_U1, t3-t1))

	D1 = np.ones((N_clusters, N_clusters))
	for i in [x for j, x in enumerate(range(len(m_new))) if j != mtr_idx]:
		NcluNclu = np.matmul(np.transpose(V[motif_idx][i]), V[motif_idx][i])
		D1 = np.multiply(D1, NcluNclu)

	D1 = np.dot(V[motif_idx][mtr_idx], D1)
	t4 = time.time()
	return (U1, D1)


def get_U2_D2(V, miu, V_two_stars, motif_idx, mtr_idx):
	m = motifs[motif_idx]
	m_new = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
	node_type = m_new[mtr_idx]
	t1 = time.time()
	cons = miu[motif_idx] / type_count_in_motifs[m][node_type]
	U2 = (1 - cons) * V_two_stars
	D2 = (1 - cons) ** 2 * V[motif_idx][mtr_idx]
	for i in range(N_motif_type):
		m_it = motifs[i]
		m_it_new = m_it
		if args.task == 'YAGO':
			m_it_new = number_remove(m_it)
		for j in range(len(m_it_new)):
			if m_it_new[j] == node_type and (i != motif_idx or j != mtr_idx):
				subtract_res = V[i][j] - V_two_stars
				pos = (np.absolute(subtract_res) + subtract_res) / 2
				neg = (np.absolute(subtract_res) - subtract_res) / 2
				U2 += cons * pos
				D2 += cons * neg
				D2 += cons**2 * V[motif_idx][mtr_idx]

	t2 = time.time()
	return (U2, D2)

def get_U3_D3(V, miu, V_two_stars, motif_idx, mtr_idx, M):
	m = motifs[motif_idx]
	m_new = m
	if args.task == 'YAGO':
		m_new = number_remove(m)
	node_type = m_new[mtr_idx]
	shape = V[motif_idx][mtr_idx].shape
	t1 = time.time()
	U3 = np.zeros(shape)
	D3 = np.zeros(shape)
	cf = miu[motif_idx] / type_count_in_motifs[m][node_type]
	D3 = cf**2 * np.multiply(M[node_type], V[motif_idx][mtr_idx]) + cf*np.multiply(M[node_type], V_two_stars)
	t2 = time.time()
	return (U3, D3)


def get_U4_D4(V, motif_idx, mtr_idx):
	m = motifs[motif_idx]
	shape = V[motif_idx][mtr_idx].shape
	U4 = np.zeros(shape)
	D4 = np.ones(shape) * lambda3
	return (U4, D4)


def get_overall_rule_V(V, motif_idx, mtr_idx, U1, D1, U2, D2, U3, D3, U4, D4):
	mtr = V[motif_idx][mtr_idx]
	mtr = np.multiply(mtr, np.sqrt((U1 + lambda1 * U2 + lambda2 * U3 + U4) / (D1 + lambda1 * D2 + lambda2 * D3 + D4)))
	return np.nan_to_num(mtr)


def get_V_three_stars(V, miu):
	# decide to not update any V_three_star after we update miu_i
	V_three_stars = []
	for i in range(N_motif_type):
		V_three_stars.append({})
		m = motifs[i]
		m_new = m
		if args.task == 'YAGO':
			m_new = number_remove(m)
		for j in range(len(m_new)):
			node_type = m_new[j]
			V_three_stars_jm = np.zeros((node_type_number[node_type], N_clusters))
			for ii in range(N_motif_type):
				mm = motifs[ii]
				mm_new = mm
				if args.task == 'YAGO':
					mm_new = number_remove(mm)
				for jj in range(len(mm_new)):
					cur_node_type = type_dictionary[ii][jj]
					if cur_node_type == node_type and ii != i:
						V_three_stars_jm += miu[ii] * V[ii][jj] / type_count_in_motifs[mm][cur_node_type]
			V_three_stars[i][node_type] = V_three_stars_jm

	return V_three_stars




def get_partial_L2_miu(V, miu, V_three_stars, motif_idx):
	partial_L2_miu = 0
	positive = 0
	negative = 0
	l = motifs[motif_idx]
	l_new = l
	if args.task == 'YAGO':
		l_new = number_remove(l)
	# below is update rule for L2 for [motif_idx]th miu

	for i in range(N_motif_type):
		m = motifs[i]
		m_new = m
		if args.task == 'YAGO':
			m_new = number_remove(m)
		for j in range(len(m_new)):
			node_type = m_new[j]
			left_in_trace = np.zeros((node_type_number[node_type], N_clusters))
			if node_type in V_three_stars[motif_idx]:
				left_in_trace = V[i][j] - V_three_stars[motif_idx][node_type]
			else:
				left_in_trace = V[i][j]

			right_in_trace = np.zeros((node_type_number[node_type], N_clusters))

			for k in range(len(l_new)):
				if l_new[k] == node_type:
					right_in_trace += V[motif_idx][k] / type_count_in_motifs[l][node_type]
			

			partial_L2_miu += -2 * trace_of_dot(left_in_trace, np.transpose(right_in_trace)) + 2 * miu[motif_idx] * np.linalg.norm(right_in_trace)**2

	return partial_L2_miu * lambda1


def get_partial_L3_miu(V, miu, M, V_three_stars, motif_idx):
	partial_L3_miu = 0
	l = motifs[motif_idx]
	l_new = l
	if args.task == 'YAGO':
		l_new = number_remove(l)

	# below is update rule for L3 for [motif_idx]th miu
	for t in node_type_number:
		sum_result = np.zeros((node_type_number[t], N_clusters))
		for i in range(len(l_new)):
			if l_new[i] == t:
				sum_result += np.multiply(M[t], V[motif_idx][i]) / type_count_in_motifs[l][t]
		partial_L3_miu += miu[motif_idx] * np.linalg.norm(sum_result)**2

		if t in V_three_stars[motif_idx]:
			partial_L3_miu += trace_of_dot(sum_result, np.transpose(np.multiply(M[t], V_three_stars[motif_idx][t])))

	return partial_L3_miu * 2 * lambda2



def get_overall_rule_miu(miu, partial_L2_miu, partial_L3_miu):
	# decide to update all miu_i and then pass into this function
	new_miu = np.zeros(miu.shape)
	partial_L_miu = partial_L2_miu + partial_L3_miu
	eta = 1.e-3  # converge condition
	new_miu = miu - eta * partial_L_miu
	new_miu = gd.simplex_project(new_miu, 0)
	return new_miu


def get_L(V, V_star, indices_list, M):
	L1 = np.zeros(N_motif_type)
	L2 = np.zeros(N_motif_type)
	L3 = 0
	for i in range(N_motif_type):
		L1[i] = getL1_sp(V, i, indices_list)
		L2[i] = getL2(V, V_star, i)
	L3 = getL3(M, V_star)
	return (L1, L2, L3)


def trace_of_dot(a, b):
	return np.sum(np.multiply(a, np.transpose(b)))


def eval_loss(V, miu, M):
	V_star = get_V_star(V, miu)
	t1 = time.time()
	L1 = 0
	L2 = 0
	L4 = 0
	for i in range(N_motif_type):
		#Xinwei edited
		m_indices_list = all_indices_list[i]
		L1 += getL1_sp(V, i, m_indices_list)
		L2 += getL2(V, V_star, i)
		L4 += getL4(V, i)
	L3 = getL3(M, V_star)
	t2 = time.time()
	print("time for calculating loss: {}".format(t2-t1))
	print("loss after update:")
	print(L1 + L2 + L3 + L4)
	return (L1, L2, L3, L4)

if __name__ == "__main__":

	#if(args.flag != 'eval'):

	print("Reading in a list of entities")
	if args.task[:4] == 'DBLP':
		with open(Config.single_author_list, 'rb') as f:
			single_author_list = pickle.load(f)
	elif args.task == 'YAGO':
		with open(Config.person_l, 'rb') as f:
			person_l = pickle.load(f)
		with open(Config.loc_list, 'rb') as f:
			loc_list = pickle.load(f)

	if DEBUG:
		print("Reading in labels")
	if args.task[:4] == 'DBLP':
		#labels_dict = label_author(seed_file)
		labels_dict,_,_ = label_author_new(labelfile_train, labelfile_test)
		author_idx_dict = {}
		for i in range(N_authors):
			author_idx_dict[single_author_list[i]] = i
		target_type_list = single_author_list
	elif args.task == 'YAGO':
		countries = Config.countries
		addr_transform = {}
		for i in range(len(countries)):
			addr_transform[countries[i]] = i+1

		'''
		with open(seed_file, "r") as fin:
			labels_dict = {}
			for line in fin:
				kv = line.split(' ')
				labels_dict[kv[0]] = addr_transform[kv[1]]
		'''
		labels_dict,_,_ = label_author_new(labelfile_train,labelfile_test)
		#print(len(labels_dict))
		#print(len(labels_dict1))
		#exit(0)
		# Naijing edited
		all_person_l = person_l
		person_idx_dict = {}
		for i in range(N_persons):
			person_idx_dict[all_person_l[i]] = i

		target_type_list = all_person_l


	label_cnt = {}
	for a in target_type_list:
		label = labels_dict[a]
		if label not in label_cnt:
			label_cnt[label] = 1
		else:
			label_cnt[label] += 1
	if DEBUG:
		print("label_cnt: {}".format(label_cnt))

	seed_label = {}
	labels_dict_test = {}

	#write a seed loader here
	_, seed_label, labels_dict_test = label_author_new(labelfile_train, labelfile_test)
		
	seed_label_cnt = {}
	for a in seed_label:
		l = seed_label[a]
		if l not in seed_label_cnt:
			seed_label_cnt[l] = 1
		else:
			seed_label_cnt[l] += 1
	
	#Xinwei edited
	labels_dict_test_cnt = {}
	for a in labels_dict_test:
		l = labels_dict_test[a]
		if l not in labels_dict_test_cnt:
			labels_dict_test_cnt[l] = 1
		else:
			labels_dict_test_cnt[l] += 1

	print('labels_dict_test_cnt:\n{}'.format(labels_dict_test_cnt))
	
	if DEBUG:
		print("seed_label_cnt: {}".format(seed_label_cnt))
		print("Use {} labels as seed".format(len(seed_label)))
		print("Use {} labels as test".format(len(labels_dict_test)))
	


	if(args.flag != 'eval'):

		M = get_M(target_type_list, seed_label)
		print("finish getting mask using seed labels")
		
		if args.task[:4] == 'DBLP':
			V = init_V_DBLP(seed_label, single_author_list)
		elif args.task == 'YAGO':
			V = init_V_YAGO(seed_label, all_person_l, seed_label_cnt, labels_dict_test_cnt, loc_list)

		miu = init_miu()
		print("initialize miu to be {}".format(miu))

		if args.task[:4] == 'DBLP':
			dict_list, dict_range_processes = prepare_T_vecs_DBLP()
		elif args.task == 'YAGO':
			dict_list, dict_range_processes = prepare_T_vecs_YAGO(labels_dict, all_person_l)

		print("finished prepare_T_vecs")

		V_star = get_V_star(V, miu)
		print("finished get_V_star")

		N_iters = 0
		#hardcoded here
		while N_iters < 2:
			print("Begin {}th overall iteration".format(N_iters))

			big_ite = 0
			while True:

				print("big_ite is {}".format(big_ite))
				p_L1, p_L2, p_L3, p_L4 = eval_loss(V, miu, M)
				for i in range(N_motif_type):
					m = motifs[i]
					m_new = m
					if args.task == 'YAGO':
						m_new = number_remove(m)
					for j in range(len(m_new)):
						print("updating motif {} matrix {}".format(i, j))
						ite = 0
						while True:
							print("{}th iteration".format(ite))
							if ite == 0:
								prev_L1, prev_L2, prev_L3, prev_L4 = eval_loss(V, miu, M)
							else:
								prev_L1, prev_L2, prev_L3, prev_L4 = L1, L2, L3, L4

							t1 = time.time()
							V_star = get_V_star(V, miu)
							V_two_stars = get_V_two_stars(V, miu, i, j, V_star)
							
							(U1, D1) = get_U1_D1(V, i, j, dict_list[i][j], dict_range_processes[i][j])
							(U2, D2) = get_U2_D2(V, miu, V_two_stars, i, j)
							(U3, D3) = get_U3_D3(V, miu, V_two_stars, i, j, M)
							(U4, D4) = get_U4_D4(V, i, j)
							
							V_old = V[i][j]
							V[i][j] = get_overall_rule_V(V, i, j, U1, D1, U2, D2, U3, D3, U4, D4)
							print("before update V[{}][{}]:".format(i, j))
							print(V_old)
							print("after update V[{}][{}]:".format(i, j))
							print(V[i][j])
							t2 = time.time()
							print("time for update rule: {}".format(t2-t1))

							delta_V = np.linalg.norm(V_old - V[i][j]) ** 2
							print("delta_V")
							print(delta_V)
							if delta_V < 0.1:
								break

							if ite % loss_eval_step_size == 0:
								(L1, L2, L3, L4) = eval_loss(V, miu, M)
							
								print("delta:")
								delta = prev_L1 - L1 + prev_L2 - L2 + prev_L3 - L3 + prev_L4 - L4
								print(prev_L1 - L1)
								print(prev_L2 - L2)
								print(prev_L3 - L3)
								print(prev_L4 - L4)
								print(delta)

								#Xinwei edited 1.e-5 -> 1.e-2
								if delta / (prev_L1 + prev_L2 + prev_L3 + prev_L4) < 1.e-8 or ite == 10:
									break

							ite += 1
							
				(L1, L2, L3, L4) = eval_loss(V, miu, M)
				big_ite += 1
				delta = p_L1 - L1 + p_L2 - L2 + p_L3 - L3 + p_L4 - L4

				#Xinwei edited 1.e-3 -> 1.e-2; 20 -> 4
				if delta / (p_L1 + p_L2 + p_L3 + p_L4) < 1.e-8 or big_ite == 10:
					break

			L1 = np.zeros(N_motif_type)
			L2 = np.zeros(N_motif_type)
			L4 = np.zeros(N_motif_type)
			V_star = get_V_star(V, miu)

			for i in range(N_motif_type):
				#Xinwei edited
				'''
				with gzip.open("intermediate/indices-list-{}.pklz".format(motifs[i]), 'rb') as f:
					m_indices_list = pickle.load(f)
				'''
				m_indices_list = all_indices_list[i]


				L1[i] = getL1_sp(V, i, m_indices_list)
				L2[i] = getL2(V, V_star, i)
				L4[i] = getL4(V, i)
			L3 = getL3(M, V_star)
			total_loss = np.sum(L1) + np.sum(L2) + L3 + np.sum(L4)


			# steps to update miu
			############################################################################
			miu_iters = 0
			while True:
				V_star = get_V_star(V, miu)
				m_L1 = 0
				m_L2 = 0
				m_L4 = 0
				for i in range(N_motif_type):
					#Xinwei edited

					m_indices_list = all_indices_list[i]


					m_L1 += getL1_sp(V, i, m_indices_list)
					m_L2 += getL2(V, V_star, i)
					m_L4 += getL4(V, i)

				m_L3 = getL3(M, V_star)

				partial_L2_miu = np.zeros(N_motif_type)
				partial_L3_miu = np.zeros(N_motif_type)

				V_three_stars = get_V_three_stars(V, miu)
				for i in range(N_motif_type):
					# need to get partial_L_miu[i]
					
					partial_L2_miu[i] = get_partial_L2_miu(V, miu, V_three_stars, i)


					partial_L3_miu[i] = get_partial_L3_miu(V, miu, M, V_three_stars, i)


				miu_old = miu
				miu = get_overall_rule_miu(miu, partial_L2_miu, partial_L3_miu)
				print("miu before update")
				print(miu_old)
				print("miu after update in {}th iteration".format(miu_iters))
				print(miu)

				delta_miu = np.linalg.norm(miu_old - miu) ** 2
				if delta_miu < 0.01:
					break

				V_star = get_V_star(V, miu)
				L1 = 0
				L2 = 0
				L4 = 0
				for i in range(N_motif_type):

					m_indices_list = all_indices_list[i]

					L1 += getL1_sp(V, i, m_indices_list)
					L2 += getL2(V, V_star, i)
					L4 += getL4(V, i)
				L3 = getL3(M, V_star)

				delta = m_L1 - L1 + m_L2 - L2 + m_L3 - L3 + m_L4 - L4
				print("delta: {}".format(delta))

				#Xinwei edited 1.e-8 -> 1.e-3; 10 -> 3
				if np.abs(delta / (m_L1 + m_L2 + m_L3 + m_L4)) < 1.e-8 or miu_iters == 30:
					break

				miu_iters += 1

			# finished miu update
			N_iters += 1


		print("After all update, need to cluster and evaluate")
		#V_star = get_V_star(V, miu)
		#print("V_star for author is:")
		#print(V_star['A'])

		
		#Xinwei edited
		if(args.flag == 'save_model'):
			f = gzip.open("saved_model.pklz", 'wb')
			V_miu = [V, miu]
			pickle.dump(V_miu, f)
			f.close()

	else:
		[V, miu] = [[], []]
		with gzip.open("saved_model.pklz", 'rb') as f:
			[V, miu] = pickle.load(f)

		
		V_star = get_V_star(V, miu)
		clusters = np.argmax(V_star['A'], axis = 1)

		#Xinwei edited
		clusters_prob = V_star['A']
		print("clusters_prob:")
		print(clusters_prob)
		clusters_prob_parse = []

		print("clusters:")
		print(clusters)
		author_label = {}
		total = 0
		correct = 0
		aal = []
		ppl = []

		#new data only
		results = {}
		ind2area = np.zeros(10)

		for i in range(len(single_author_list)):
			a = single_author_list[i]
			predict = clusters[i] + 1
			author_label[a] = predict
			if a in labels_dict_test:
				total += 1
				actual = labels_dict_test[a]
				if actual == predict:
					correct += 1
				aal.append(actual)
				ppl.append(predict)
				results[a] = (actual, predict)
				clusters_prob_parse.append(clusters_prob[i])

		print('results is: {}'.format(results))
		print("Total test size: {}".format(total))
		try:
			print("Accuracy: {}".format(correct / total))
		except ZeroDivisionError:
			print("0 total labels")

		precision = precision_score(np.array(aal), np.array(ppl), average = None)
		micro_precision = precision_score(np.array(aal), np.array(ppl), average='micro')
		macro_precision = precision_score(np.array(aal), np.array(ppl), average='macro')
		recall = recall_score(np.array(aal), np.array(ppl), average = None)
		micro_recall = recall_score(np.array(aal), np.array(ppl), average='micro')
		macro_recall = recall_score(np.array(aal), np.array(ppl), average='macro')
		f1 = f1_score(np.array(aal), np.array(ppl), average=None)
		micro_f1 = f1_score(np.array(aal), np.array(ppl), average='micro')
		macro_f1 = f1_score(np.array(aal), np.array(ppl), average='macro')
		nmi = normalized_mutual_info_score(np.array(aal), np.array(ppl))
		ll = log_loss(np.array(aal), clusters_prob_parse)

		print('len(np.array(aal)) is: {}'.format(len(np.array(aal))))
		print('len(clusters_prob) is: {}'.format(len(clusters_prob)))
		print('len(clusters_prob_parse) is: {}'.format(len(clusters_prob_parse)))
		print("precision:")
		print(precision)
		print('micro precision: {}'.format(micro_precision))
		print('macro precision: {}'.format(macro_precision))
		print("recall:")
		print(recall)
		print('micro recall: {}'.format(micro_recall))
		print('macro recall: {}'.format(macro_recall))
		print("f1:")
		print(f1)
		print('micro f1: {}'.format(micro_f1))
		print('macro f1: {}'.format(macro_f1))
		print("nmi:")
		print(nmi)
		print("ll")
		print(ll)




