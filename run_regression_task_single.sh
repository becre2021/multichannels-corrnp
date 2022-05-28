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

#not good for allowing more randomness for sampling of gumbel 
# python3 test_syntheticdataset_metatask_single.py --modelname gpdep --tasktype singletask --dep --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 11 --msg $msg;
# python3 test_syntheticdataset_metatask_single.py --modelname gpdep --tasktype singletask --dep --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 11 --msg $msg;

#python3 test_syntheticdataset_metatask_single.py --modelname base --tasktype singletask --dep --cnntype deep  --nepochs $nepochs --lr 0.0005 --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;
# python3 test_syntheticdataset_metatask_single.py --modelname baselatent --tasktype singletask --dep --cnntype deep  --nepochs $nepochs --lr 0.0005 --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;
# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;

# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;

# python3 test_syntheticdataset_metatask_single.py --modelname gpdep --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;


# # 5/18 to check rbf prior 
# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2 --msg $msg;

# # 5/18 to check sm prior int [0,5] 
# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 3 --msg $msg;

#5/18 to check anp - re
python3 test_syntheticdataset_metatask_single.py --modelname anp --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;

# 5/18 to check cnp - re
python3 test_syntheticdataset_metatask_single.py --modelname cnp --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;



#------------------------------------------------------------------
# large context set setting
#------------------------------------------------------------------
datav=2
msg=mixedtask/lessrandomness/tempring1.0

# #python3 test_syntheticdataset_metatask_single.py --modelname base --tasktype singletask --dep --cnntype deep  --nepochs $nepochs --lr 0.0005 --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;
# python3 test_syntheticdataset_metatask_single.py --modelname gpdep --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;
# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;

# python3 test_syntheticdataset_metatask_single.py --modelname gpdep --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;


# # 5/18 to check rbf prior 
# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 2 --msg $msg;

# # 5/18 to check sm prior int [0,5] --> quite good results when data large
# python3 test_syntheticdataset_metatask_single.py --modelname gpind --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 3 --msg $msg;

#5/18 to check anp - re
python3 test_syntheticdataset_metatask_single.py --modelname anp --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;

# 5/18 to check cnp - re
python3 test_syntheticdataset_metatask_single.py --modelname cnp --tasktype singletask --dep --cnntype deep  --nepochs $nepochs  --lr 0.0005 --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 1 --msg $msg;
