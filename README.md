# MoCHIN

This repository provides code and data for the paper:<br>
> User-Guided Clustering in Heterogeneous Information Networks via Motif-Based Comprehensive Transcription<br>
> Yu Shi*, Xinwei He*, Naijing Zhang*, Carl Yang, and Jiawei Han.<br>
> Submitted to ECMLPKDD 2019.<br>

Particularly, it includes 
(1) an implementation of MoCHIN model, 
(2) the partial DBLP dataset and the partial YAGO dataset (full datasets are excluded from this repository due to file sizes), and 
(3) the class labels used in the DBLP and YAGO clustering tasks.

## Basic Usage
	$ python MoCHIN.py --task $name_of_the_task [--save_model $save_model_path --eval $file_to_be_evaluated —-debug]

where the arguments in the brackets are optional.

#### Input

--task TASK:	DBLP_AREA, DBLP_GROUP, YAGO, $SELF_DEFINED

--save_model SAVE_MODEL_PATH	run our model and save parameters to the specified path

--eval FILE_NAME	run evaluation on the specified file

--debug	control ouput verbosity

e.g.

	$ python3 MoCHIN.py --task DBLP_AREA --save_model model --debug
	


Please provide a config.py if running self-defined task.


### Input Data
Make a folder in MoCHIN with name: “input_" + $task

Inside the constructed folder, for each motif that is going to be used, construct a indices-list-$motif.pklz file

indices-list-$motif.pklz: a list of tuples that encodes motif instance in a predefined order, where each node in a motif instance is represented by its pre-assigned index. For each motif encode by the tuple, each node with the same node type in a motif instance with node index permutatd inside a tuple representation are considered as differenc motif instances.

e.g.
	(a1,p1,p2,a2), (a1,p2,p1,a2), (a2,p1,p2,a1) and (a2,p2,p1,a1) are considered as 4 different motif instances
the list is stored in indices-list-$motif.pklz in a binary format using pickle and gzip


node-type-number.pklz: a dictionary with key representing node types in the input graph network and values representing number of a specific node type in the data
e.g.
	{'P': 11138, 'V': 1564, 'A': 245, 'T': 6550}

the list is stored in node-type-number.pklz in a binary format using pickle



make a sub-folder "tran_test_split_$task", where $task-train-label.txt and $task-test-label.txt stores the label for training date and testing data respectively in the following format :

	Instance1	label1
	Instance2	label2

	...
	
	InstanceN	labelN



### Original Data

entity-dblp-subsample: subsample ...

$nodetype-list.pklz: a pickle list of nodes of $nodetype(e.g.author)

label-$task.txt: a combination of $task-train-label.txt and $task-test-label.txt



### Eval Data
The input file of --eval is a .tsv file (tab seperated values) where each line represents the distribution of weights of a target instance for each cluster, where the weights are seperated by '\t' 

Notice that:

	1. the sum of weights in each line not necessarily sums to 1)
	2. the order of target instance and groups are pre-specified in single-author-list.pklz in Input Data
	3. as reflected in single-author-list.pklz the group weight distribution of training data should also be included in the .tsv file


### Saved Model
saved_model can be send as input to MoCHIN.py and evaluated without running MoCHIN model by using eval flag.

saved_model: a list l with 2 parameter l[0] = V， l[1] = miu s.t. l is stored in saved_model in a binary format using pickle and gzip.

V: a 3-dimensional nested list s.t. V[i] representes i-th node type in a motif. V[i] is a 2-dimensional nested list with dimension #node with type i x #cluster, in which #cluster is a pre-defined parameter. for seed node with type i1, index j1, label k V[i1][j1][k] = 1 and V[i1][j1][l] = 0 for l != k, for non-seed node with type i2, index j2 V[i2][j2][l] = (1/#cluster)**(0.5) for all l.

miu: list of initial weight for each motif in a pre-defined order (default to be uniform)


### Miscellaneous

Please send any questions you might have about the codes and/or the algorithm to <xhe17@illinois.edu>, <nzhang31@illinois.edu>, <yushi2@illinois.edu>.

