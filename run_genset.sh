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






# tasktype=sin3
# SET=$(seq 51 200)
# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;            
#     python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
# done

# tasktype=mosm
# SET=$(seq 51 200)
# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;            
# done

# for i in $SET
# do
#     python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
# done



# tasktype=sin3
# # #SET=$(seq 51 200)
# # SET=$(seq 1 400)
# # for i in $SET
# # do
# #     python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra --train --num_saved $i;           
# #     python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra --train --num_saved $i;                
# # done
# # python3 dataset_generate_setdict.py --tasktype $tasktype --dep --testtype extra  --num_saved -1;           
# # python3 dataset_generate_setdict.py --tasktype $tasktype  --testtype extra       --num_saved -1;                
# python3 generate_synthetic_setdict.py --tasktype $tasktype --dep --testtype extra  --num_saved -1;           
# python3 dataset_synthetic_setdict.py --tasktype $tasktype  --testtype extra       --num_saved -1;                



# #tasktype=sin3
# SET=$(seq 1 100)
# for i in $SET
# do
#     #python3 generate_real_airquality.py --num_saved $i;           
#     python3 generate_real_waterdepth.py --num_saved $i;           
    
# done

#generate_synthetic_setdict_v2



# tasktype=sin3
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask 2000 --nbatch 32 --dep;            
# python3 generate_synthetic_setdict_v2.py --tasktype $tasktype --testtype extra --ntask 2000 --nbatch 32 ;            














# python3 dataset_generate_setdict.py --tasktype $tasktype  --dep  --testtype extra --num_saved -1;
# python3 dataset_generate_setdict.py --tasktype $tasktype   --testtype extra --num_saved -1;
















#python3 generate_setdict.py --num_saved -1 --train trainno --testtype extra;
#python3 generate_setdict.py --tasktype lmc --dep 'testtype' extra --train





