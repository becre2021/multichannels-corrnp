#!/bin/bash


# #tasktype=sin3
# #tasktype=lmc
# tasktype=mosm
# SET=$(seq 1 50)
# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;            
#     python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
# done
# python3 dataset_generate_setdict.py --tasktype $tasktype  --dep  --testtype extra --num_saved -1;
# python3 dataset_generate_setdict.py --tasktype $tasktype   --testtype extra --num_saved -1;






#tasktype=sin3
tasktype=lmc
#SET=$(seq 51 200)
SET=$(seq 201 400)
for i in $SET
do
    python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;          
    python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
done


# nepochs=400
# for l_scale in "0.1"; do
# # for l_scale in "0.01"; do
# # for l_scale in "0.001"; do
# #for l_scale in "0.1" "0.01" "0.001"; do
#     #python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm          --nepochs $nepochs  --initl $l_scale --cnntype deep ;
    
#     #python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;     
    
#     python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
    
# done    



# tasktype=lmc
# SET=$(seq 51 200)
# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;            
# done

# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
# done



# tasktype=mosm
# SET=$(seq 51 200)
# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;            
#     python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
# done



# python3 dataset_generate_setdict.py --tasktype $tasktype  --dep  --testtype extra --num_saved -1;
# python3 dataset_generate_setdict.py --tasktype $tasktype   --testtype extra --num_saved -1;
















#python3 generate_setdict.py --num_saved -1 --train trainno --testtype extra;
#python3 generate_setdict.py --tasktype lmc --dep 'testtype' extra --train





