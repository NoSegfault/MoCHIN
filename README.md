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
	$ python3 MoCHIN.py $lambda1_$lambda2_$lambda3 $m1_$m2_..._$mk $seed-ratio $task $flag

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

	$ python3 MoCHIN.py 1_10_0.001 0.25_0.25_0.25_0.25 0.01 DBLP_GROUP save_model

Please provide a config.py if running self-defined task.


### Miscellaneous

Please send any questions you might have about the codes and/or the algorithm to <yushi2@illinois.edu>.

