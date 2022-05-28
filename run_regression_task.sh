#!/bin/bash


#tasktype=mosm
nepochs=500
#datav=9
datav=11

# tasktype=lmc
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask $nepochs --nbatch 16  --train  --datav $datav  --dep ;           
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra  --ntask $nepochs --nbatch 16 --train  --datav $datav ;            


# tasktype=mosm
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask $nepochs --nbatch 16  --train  --datav $datav  --dep ;           
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra  --ntask $nepochs --nbatch 16 --train  --datav $datav ;            



# tasktype=mosmvarying
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask $nepochs --nbatch 16  --train  --datav $datav  --dep ;           
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra  --ntask $nepochs --nbatch 16 --train  --datav $datav ;            


# tasktype=sin3
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask $nepochs --nbatch 16   --train  --datav $datav  --dep ;            
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra  --ntask $nepochs --nbatch 16  --train  --datav $datav ;            





# tasktype=mosm
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask $nepochs --nbatch 16  --train  --datav $datav  --dep ;           
# tasktype=mosmvarying
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask $nepochs --nbatch 16  --train  --datav $datav  --dep ;           
















# # # # # priorscale 0.5 teste
# # # # # --> when shallow net is used, the perfomrance are degraded 
# # # # # --> using resstuructre improve perfomrance a liittle bit.
# # # # #--------------------------------------

# msg=shallownetcheck
# msg=strongdependecyset
# msg=smallset
# msg=morerandomness
# msg=datadrivenprior
# msg=sin3depfalsenew
# msg=newdeptest
# msg=spikeslabprior
# msg=spikeslabprior/priorlearningsteplr
# msg=spikeslabprior/priorlearningsteplr/neural #v50
# msg=spikeslabprior/priorlearningsteplr50/neural/reducerandomness/ #v50
msg=spikeslabprior/priorlearningsteplr/neural-nummixture10-tmpering.1/ #v52
msg=spikeslabprior/neural-nummixture10-tmpering.1/ #v52

#datav=9
#datav=10
datav=11

nepochs=500
printfreq=20




# # # #------------------------------------------------------------------
# # # #depedency true
# # # #------------------------------------------------------------------

tasktype=mosm
# # # #python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# # # #python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.01 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;

# # # # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# # # #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 26 --msg $msg;
# # # # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# # # # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 26 --msg $msg;

# # # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 27 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 27 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 27 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdind --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;



# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdind --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 44 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 46 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 47 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 48 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 49 --msg $msg;


# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 50 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 51 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 52 --msg $msg;






# python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;
# python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;

# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;
# python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;
# python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 61 --msg freqmean[0,5] --dep;





tasktype=sin3
# #python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.01 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 26 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 26 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 27 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 27 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 27 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 30 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 30 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 30 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.5 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 30 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.5 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 30 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 32 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 33 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 34 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.1 --runv 35 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 36 --msg $msg;


# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;


# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 44 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 48 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 49 --msg $msg;

# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 50 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 51 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 52 --msg $msg;


# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep   --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep   --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep   --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;


# python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep   --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;


# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep   --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg;



python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep   --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 61 --msg rbftprior;
python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 61 --msg rbftprior;




#-----------------------------------------------------------------------
tasktype=mosmvarying
#-----------------------------------------------------------------------

# # # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;
# # #python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.01 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 25 --msg $msg;

# # #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 26 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 26 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 26 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 26 --msg $msg;

# # # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 27 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 27 --msg $msg;
# # # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype shallow  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 27 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 30 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 30 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 30 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 32 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 1.0 --runv 34 --msg $msg;




# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;

# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 44 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 48 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 49 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 50 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 50 --msg $msg;


# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 51 --msg $msg;
# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 52 --msg $msg;


# python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;
# python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;

# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;
# python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 60 --msg $msg --dep;
# python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 61 --msg freqmean[0,5] --dep;








#-----------------------------------------------------------------------
tasktype=lmc
#-----------------------------------------------------------------------

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep --dep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname base --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname baselatent --tasktype $tasktype --cnntype deep  --nepochs $nepochs --initl 0.1 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 41 --msg $msg;


# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 42 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 43 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 44 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpind --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 45 --msg $msg;

# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 48 --msg $msg;
# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 49 --msg $msg;


# #python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 50 --msg $msg;
# # python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 51 --msg $msg;


# python3 test_syntheticdataset_metatask.py --modelname gpdep --tasktype $tasktype --cnntype deep  --dep --nepochs $nepochs --initl 1.0 --datav $datav --printfreq $printfreq --reglambda 0.0 --runv 52 --msg $msg;



