#!/bin/bash




#------------------------------------------------------------------------
# dataset generations
#------------------------------------------------------------------------

nepochs=1000

#datav=1
#datav=2
#python3 generate_synthetic_setdict_v3.py  --ntask $nepochs --nbatch 16  --train  --datav $datav --dep ;           

















# #------------------------------------------------------------------------
# # training alorightm
# #------------------------------------------------------------------------

msg=mixedtask #v52
nepochs=500
#datav=1
printfreq=20






#------------------------------------------------------------------
# small context set setting
#------------------------------------------------------------------
datav=1
msg=mixedtask/lessrandomness/tempring1.0


#------------------------------------------------
# 5/22 to check ablation study for mixture Q
#------------------------------------------------

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 2 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 4 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 5 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;





#---------------------------------------------------------------------------
# 5/23 to check ablation study for mixture Q; previous expeirmetn decresing pefromance before 
# so needed to check when we obtain superior performances
#---------------------------------------------------------------------------

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 2 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 4 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 5 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;





#------------------------------------------------------------------
# large context set setting
#------------------------------------------------------------------
datav=2
msg=mixedtask/lessrandomness/tempring1.0

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 2 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 4 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 5 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;






#---------------------------------------------------------------------------
# 5/23 to check ablation study for mixture Q; previous expeirmetn decresing pefromance before 
# so needed to check when we obtain superior performances
#---------------------------------------------------------------------------

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 2 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 4 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 5 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

