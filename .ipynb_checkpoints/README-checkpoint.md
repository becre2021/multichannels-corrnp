<!-- ![Demonstration of a ConvCNP](https://github.com/cambridge-mlg/convcnp/blob/master/demo_images/convcnp.gif) -->



##  Bayesian Convolutional Deep Sets with Task-Dependent Stationary Prior

We provide the implementation and experiment results for the paper: Bayesian Convolutional Deep Sets with Task-Dependent Stationary Prior

<p align="center">
    <img src="https://github.com/becre2021/multichannels-corrnp//img/concept.pdf" width="500" height="250">
</p>

 
## Description

### Proposed Methodology

* models/test_gpsampler7.py : random data representation using the task-dependent stationary prior  in Eq. (10) (main)
* models/test_dep_correlatenp.py : proposd NP model using a Bayesian Convolutional deep sets 
* train_synthetic_task_single.py : train models for section 5.1 experiment
* train_synthetic_task_multi.py : train models for section 5.2 experiment


### Experiments
* examples_construct_representation.ipynb : examples of random functional representation
* examples_dataset_single.ipynb : examples of tasks for section 5.1 experiment
* examples_task_single.ipynb : experiment results for section 5.1 experiment
* examples_task_multi.ipynb : experiment results for section 5.2 experiment


## Requirements

* python >= 3.6
* torch = 1.7
* pandas
* scipy
* attrdict


## Installation

    git clone https://github.com/becre2021/multichannels-corrnp.git
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 


## Reference 

* https://github.com/cambridge-mlg/convcnp
* https://github.com/makora9143/pytorch-convcnp 
* https://github.com/GAMES-UChile/mogptk/
* https://github.com/juho-lee/bnp/tree/master/regression
* https://github.com/j-wilson/GPflowSampling



