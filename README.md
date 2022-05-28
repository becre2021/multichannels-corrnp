<!-- ![Demonstration of a ConvCNP](https://github.com/cambridge-mlg/convcnp/blob/master/demo_images/convcnp.gif) -->



##  Bayesian Convolutional Deep Sets with Task-Dependent Stationary Prior

We provide the implementation and experiment results for the paper: Bayesian Convolutional Deep Sets with Task-Dependent Stationary Prior


 
## Description

### Methods

<!-- * models/gp_rrff_reg.py : a sampling based variational inference (SVSS)
* models_utility/spt_manager_train.py : a sampling strategy of Propoistion 2 
* models_utility/personalized_adam.py : an approximate natural gradient update of Proposition 3
 -->
### Experiments

<!-- * exp1_main_section5-1.ipynb, exp2_main_section5-1.ipynb : validation of the weight sampling that reduces the error of ELBO estimator (main-expeiriment section 5.1)
* main_ablationstudy.py : ablation study for SVSS (main-experiment section 5.2)
* main_uci_regression.py, main_uci_regression_batch.py : a large-scale and high-dimensional UCI dataset regression (main-experiment section 5.3)
* exp1_appendix.ipynb : validation of the scalable weight sampling (supplementary-experiment section 4.1)
* SM kernel Recovery by SVSS-Ws.ipynb : SM kernel Recovery conducted in [experiment section 5.1](https://arxiv.org/pdf/1910.13565.pdf)
 -->

## Requirements

* python >= 3.6
* torch = 1.7
* pandas
* scipy


## Dataset

<!-- * datasets/uci_datasets/ : kin8nm and parkinsons set used in our experiment
* datasets/uci_wilson/ : download [UCI Wilson dataset](https://drive.google.com/file/d/0BxWe_IuTnMFcYXhxdUNwRHBKTlU/view) and unzip the downloaded file
 -->

## Installation

    git clone https://github.com/ABInferGSM/src.git
    if necessary, install the required module as follows
    pip3 install module-name
    ex) pip3 install numpy 


## Reference 

* https://github.com/cambridge-mlg/convcnp
* https://github.com/makora9143/pytorch-convcnp 
* https://github.com/GAMES-UChile/mogptk/
* https://github.com/juho-lee/bnp/tree/master/regression



