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
	$ python3 MoCHIN.py $task $flag

#### Input

$lambda1: regularization1 weight
$lambda2: regularization2 weight
$lambda3: regularization3 weight

$m1: motif1 weight
$m2: motif2 weight
$mk: motifk weight

task: DBLP_AREA, DBLP_GROUP, YAGO_GROUP, $SELF_DEFINED

flag: save_model	run our model and save parameers to saved_model.pklz
      eval:	evaluation the saved_model.pklz

e.g.

	$ python3 MoCHIN.py DBLP_GROUP save_model

Please provide a config.py if running self-defined task.


### Input Data
make a folder in MoCHIN with name: “input_" + $task
inside folder for each motif that are planned to use construct a indices-list-$motif.pklz file

indices-list-$motif.pklz: a list of tuples that encodes motif instance in a predefined order where each node in a motif instance is represented by its pre-assigned index. For each motif encode by the tuple, each node with the same node type in a motif instance with node index permutatd inside a tuple representation are considered as differenc motif instances.
e.g.
	(a1,p1,p2,a2), (a1,p2,p1,a2), (a2,p1,p2,a1) and (a2,p2,p1,a1) are considered as 4 different motif instances
the list is stored in indices-list-$motif.pklz in a binary format using pickle and gzip

### Saved Model
saved_model can be send as input to MoCHIN.py and evaluated without running MoCHIN model by using eval flag.

saved_model: a list l with 2 parameter l[0] = V， l[1] = miu s.t. l is stored in saved_model in a binary format using pickle and gzip.

V: a 3-dimensional nested list s.t. V[i] representes i-th node type in a motif. V[i] is a 2-dimensional nested list with dimension #node with type i x #cluster, in which #cluster is a pre-defined parameter. for seed node with type i1, index j1, label k V[i1][j1][k] = 1 and V[i1][j1][l] = 0 for l != k, for non-seed node with type i2, index j2 V[i2][j2][l] = (1/#cluster)**(0.5) for all l.

miu: list of initial weight for each motif in a pre-defined order (default to be uniform)


### Miscellaneous

Please send any questions you might have about the codes and/or the algorithm to <xhe17@illinois.edu>, <nzhang31@illinois.edu>, <yushi2@illinois.edu>.

