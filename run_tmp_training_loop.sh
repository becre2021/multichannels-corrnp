#!/bin/bash

# -----------------------------------------------------------------
# this code was used to run old code 
# but, this could be employsed to make forloop code for ablation study (for referecens)
#
# -----------------------------------------------------------------









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
#done



#nepochs=100
# for l_scale in "1.0" "0.1" "0.01"; do

#     python3 test_training_correlatenp.py --tasktype sin3 --dep --nepochs $nepochs --initl $l_scale; 
#     python3 test_training_correlatenp.py --tasktype lmc --dep --nepochs $nepochs  --initl $l_scale; 
#     python3 test_training_correlatenp.py --tasktype mosm --dep --nepochs $nepochs --initl $l_scale;
# done


# nepochs=100
# for l_scale in "1.0" "0.1" "0.01"; do
#     python3 test_training_correlatenp.py --tasktype sin3  --nepochs $nepochs  --initl $l_scale; 
#     python3 test_training_correlatenp.py --tasktype lmc   --nepochs $nepochs  --initl $l_scale; 
#     python3 test_training_correlatenp.py --tasktype mosm  --nepochs $nepochs  --initl $l_scale;
# done


#nepochs=200
#for l_scale in "0.1"; do
#for l_scale in "0.01"; do
#for l_scale in "0.001"; do
# for l_scale in "0.1" "0.01" "0.001"; do
#     python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm          --nepochs $nepochs  --initl $l_scale --cnntype deep ;
    
#     python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ;
    
# done


#nepochs=200
#nepochs=50

# python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl 0.01 --cnntype deep ; 
# python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype deep ; 

#python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl 1.0 --cnntype deep ; 
#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype deep ; 
#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype deep ; 
#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.01 --cnntype deep ; 
#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype deep ; 

#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype shallow ; 


# nepochs=200
# for l_scale in "0.1"; do
# # for l_scale in "0.01"; do
# # for l_scale in "0.001"; do
# #for l_scale in "0.1" "0.01" "0.001"; do
#     #python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm          --nepochs $nepochs  --initl $l_scale --cnntype deep ;
    
#     #python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ; 
#     python3 test_training_correlatenp.py --tasktype mosm --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ;    
#     #python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
# done    


nepochs=400
#for l_scale in "0.1"; do
#for l_scale in ".5"; do
#for l_scale in ".1"; do
# for l_scale in "1."; do

# # for l_scale in "0.01"; do
# # for l_scale in "0.001"; do
# #for l_scale in "0.1" "0.01" "0.001"; do
#     #python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm          --nepochs $nepochs  --initl $l_scale --cnntype deep ;
    
#     #python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm --dep    --nepochs $nepochs --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;     
    
# #     python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
# #     python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
    
    
#     #python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype sin3 --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
    

#     python3 test_training_correlatenp.py --tasktype sin3           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     python3 test_training_correlatenp.py --tasktype sin3 --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 

#     python3 test_training_correlatenp.py --tasktype mosm           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 

# done    

# l_scale="0.1"
# python3 test_training_correlatenp.py --tasktype sin3 --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
# python3 test_training_correlatenp.py --tasktype mosm           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
# python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 


# l_scale=".1"
# python3 test_training_correlatenp.py --tasktype sin3 --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;
l_scale=".1"
python3 test_training_correlatenp.py --tasktype sin3 --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;
python3 test_training_correlatenp.py --tasktype sin3           --nepochs $nepochs  --initl $l_scale --cnntype deep ;
python3 test_training_correlatenp.py --tasktype lmc  --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;
python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ;
python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;
python3 test_training_correlatenp.py --tasktype mosm           --nepochs $nepochs  --initl $l_scale --cnntype deep ;


# l_scale="1."
# python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;
# l_scale=".1"
# python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ;
# hours taken monday 9 am checked 
# check weak validatin effective 








#python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
# 6hours
# ;



# for l_scale in "1."; do
#     python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep & 
#     python3 test_training_correlatenp.py --tasktype lmc --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep & 
    
# done    
# ;
# for l_scale in "1."; do
#     python3 test_training_correlatenp.py --tasktype mosm           --nepochs $nepochs  --initl $l_scale --cnntype deep & 
#     python3 test_training_correlatenp.py --tasktype mosm --dep     --nepochs $nepochs  --initl $l_scale --cnntype deep & 
    
# done    
# ;
