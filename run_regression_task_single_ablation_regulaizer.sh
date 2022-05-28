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



# #---------------------------------------------------------------------------
# # 5/24 to check how varying lambda affects the training results; (reported results are with 0.5)
# # runv==2, [0,10hz]

# #---------------------------------------------------------------------------
# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 1  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.5 --runv 1  --msg $msg;



#---------------------------------------------------------------------------
# 5/24 to check how varying lambda affects the training results; (reported results are with 0.5)
# runv==1, [0,5hz]

#---------------------------------------------------------------------------
# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 1  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.5 --runv 1  --msg $msg;



# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.1 --runv 1  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.25 --runv 1  --msg $msg;



#---------------------------------------------------------------------------
# 5/25 to check how varying lambda affects the training results; (reported results are with 0.5)
# runv==3, [0,5hz],  self.tempering0 = 1e-2
#---------------------------------------------------------------------------

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 3  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.1 --runv 3  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.5 --runv 3  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 3  --msg $msg;


# # ---------------------------------------------------------------------------
# # 5/25 to 1e-1 for pseduo p_gp unlike orignal 1e0 for pseudo gp
# # runv==3, [0,5hz],  self.tempering0 = 1e-1
# # ---------------------------------------------------------------------------

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.5 --runv 1  --msg $msg;



# # ---------------------------------------------------------------------------
# # 5/25  runv==3, [0,5hz],  self.tempering0 = 5e-2, change frequency init param 
# # ---------------------------------------------------------------------------

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.5 --runv 3  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 3  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 3  --msg $msg;



# ---------------------------------------------------------------------------
# 5/25  runv==3, [0,5hz],  self.tempering0 = 5e-2, change frequency init param 
# ---------------------------------------------------------------------------

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.5 --runv 4  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 4  --msg $msg;

python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 4  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 4  --msg $msg;














#------------------------------------------------------------------
# large context set setting
#------------------------------------------------------------------
# datav=2
# msg=mixedtask/lessrandomness/tempring1.0

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 2 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 4 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 5 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;


# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1  --msg $msg;






#---------------------------------------------------------------------------
# 5/23 to check ablation study for mixture Q; previous expeirmetn decresing pefromance before 
# so needed to check when we obtain superior performances
#---------------------------------------------------------------------------

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 2 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 3 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 4 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

# python3 test_syntheticdataset_metatask_single_ablation.py --modelname gpdep --nmixtures 5 --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2  --msg $msg;

