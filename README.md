# Experiment Materials
This repository stores the experiments materials from the paper "Selecting
Algorithms for the Quadratic Assignment Problem with a Multi-label Meta-learning
Approach" to be pusblished at the Proceedings of the [The 7th Brazilian
Conference on Intelligent Systems
(BRACIS)](https://bracis2018.mybluemix.net/index.html).


The content of each folder is described next.

### Algorithms
This folder contains the source-codes of the meta-heuristics with their default
parameters. It is also provided the set of seeds used for the experiments and
the runner scripts for the landscape features extraction. Besides, the fitted
function used to determine the execution time based on the instance size can be
found here.

### Algorithms Performances
The results achieved by the algorithms over 30 runs on each intance.

### Dataset
The generated multi-label classification dataset. The labels are integer
encoded and must be binarized before using them with the scikit-lean methods.

### Reults
It contains the results metrics, such as the Accuracy and F-Scores, of both
techniques performed for handling the multi-label classification.

### QAP Intances
These are all the instances retrieved from
[QAPLIB](http://anjos.mgi.polymtl.ca/qaplib/inst.html), along with their
respective best known solutions.
