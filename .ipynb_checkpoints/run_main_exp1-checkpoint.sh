#!/bin/bash

# kernels=("eq" "matern" "noisy-mixture" "weakly-periodic" "sawtooth")
# for data in "${kernels[@]}"; do
#     python train.py $data cnp --train --root _experiments/cnp-$data --learning_rate 3e-4
#     python train.py $data anp --train  --root _experiments/anp-$data --learning_rate 3e-4
#     python train.py $data convcnp --train --root _experiments/convcnp-$data --learning_rate 3e-4
#     python train.py $data convcnpxl --train --root _experiments/convcnpxl-$data --learning_rate 1e-3
# done


# datasets="sin3 mosm lmc"
# #datasets="lmc"
# for datas in $datasets
# do
#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model cnp --nbasis 3 --trainloss 'weight' --nepochs 50
#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model cnp --nbasis 3 --trainloss 'uniform' --nepochs 50

#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model anp --nbasis 3 --trainloss 'weight' --nepochs 50
#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model anp --nbasis 3 --trainloss 'uniform' --nepochs 50

#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model convcnp --nbasis 3 --trainloss 'weight' --nepochs 50
#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model convcnp --nbasis 3 --trainloss 'uniform' --nepochs 50
    
#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model propose --nbasis 3 --trainloss 'weight' --nepochs 50
#     python3 main_exp1.py --data $datas --testtype inter --nchannel 3 --model propose --nbasis 3 --trainloss 'uniform' --nepochs 50
# done


#python3 multitask_kernel_v7_datageneration_withpaper_matern_lmc.py;
# python3 main_exp1-synthetic.py --data mosm --dep;
# python3 main_exp1-synthetic.py --data sin3 --dep;
#python3 main_exp1-synthetic.py --data lmc --dep;


#main_exp1_synthetic_v2.py

#python3 main_exp1_synthetic_v2.py --data mosm --dep;
#python3 main_exp1_synthetic_v2.py --tasktype sin3 --dep --nepochs 400 ;
# python3 main_exp1_synthetic_v2.py --data lmc --dep;
        #main_exp1_synthetic_v2.py
#main_exp1_synthetic_v2.py



python3 main_exp1_synthetic_v2.py --tasktype sin3 --dep --nepochs 400 ;