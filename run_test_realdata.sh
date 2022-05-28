#!/bin/bash





#tasktype=sin3
# SET=$(seq 1 500)
# for i in $SET
# do
#     python3 generate_real_airquality.py --num_saved $i;           
# done














# nepochs=500
# #for l_scale in "0.1" "0.05"; do
# #for l_scale in "0.5"; do
# for l_scale in "0.1"; do
# #for l_scale in "0.1"; do
# #for l_scale in "1."; do
#     python3 test_realdataset_airquality_metatask_v2.py  --modelname base       --initl $l_scale --nepochs $nepochs ;
#     #python3 test_realdataset_airquality_metatask_v2.py  --modelname baselatent --initl $l_scale --nepochs $nepochs ;
#     python3 test_realdataset_airquality_metatask_v2.py  --modelname gpind      --initl $l_scale --nepochs $nepochs ;
#     #python3 test_realdataset_airquality_metatask_v2.py  --modelname gpdep      --initl $l_scale --nepochs $nepochs ;
# done
# for l_scale in "0.1" "0.01" "0.001"; do
#     python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype lmc           --nepochs $nepochs  --initl $l_scale --cnntype deep ; 
#     #python3 test_training_correlatenp.py --tasktype mosm          --nepochs $nepochs  --initl $l_scale --cnntype deep ;



# SET=$(seq 1 500)
# for i in $SET
# do
#     python3 generate_real_airquality.py --num_saved $i;               
# done







# dataname='airquality'
# #SET=$(seq 1 2000)
# SET=$(seq 1 20)

# for i in $SET
# do
#     python3 generate_real_airquality.py --num_saved $i --datav=$datav;               
# # done











# datav=2
# dataname='airquality'
# python3 generate_real_airquality.py --num_totalsaved 1000 --datav=$datav;               






# nepochs=500
# for l_scale in "0.1" "0.05"; do
# for l_scale in "0.5"; do
# for l_scale in "0.1"; do
# for l_scale in "0.2"; do


dataname='airquality'
nepochs=1000
datav=2

#for l_scale in "0.05" "0.01"; do
# #for l_scale in "1."; do
# for l_scale in "0.1"; do
#     python3 test_realdataset_metatask.py  --modelname base    --dataname $dataname   --datav $datav  --initl $l_scale --nepochs $nepochs ;
#     python3 test_realdataset_metatask.py  --modelname gpind    --dataname $dataname   --datav $datav  --initl $l_scale --nepochs $nepochs ;
#     python3 test_realdataset_metatask.py  --modelname gpdep   --dataname $dataname    --datav $datav  --initl $l_scale --nepochs $nepochs ;
#     python3 test_realdataset_metatask.py  --modelname gpdep2   --dataname $dataname   --datav $datav  --initl $l_scale --nepochs $nepochs ;
    
#     #python3 test_realdataset_airquality_metatask_v2.py  --modelname gpdep      --initl $l_scale --nepochs $nepochs ;
# done


python3 test_realdataset_metatask.py  --modelname gpind   --dataname $dataname    --datav $datav  --initl 0.1 --nepochs $nepochs ;
python3 test_realdataset_metatask.py  --modelname gpdep    --dataname $dataname    --datav $datav  --initl 0.1 --nepochs $nepochs ;

#python3 test_realdataset_metatask.py  --modelname gpind   --dataname $dataname    --datav $datav  --initl 0.1 --nepochs $nepochs ;
# python3 test_realdataset_metatask.py  --modelname gpdep2   --dataname $dataname    --datav $datav  --initl 0.5 --nepochs $nepochs ;
#python3 test_realdataset_metatask.py  --modelname base   --dataname $dataname    --datav $datav  --initl 0.01 --nepochs $nepochs ;
# python3 test_realdataset_metatask.py  --modelname baselatent   --dataname $dataname    --datav $datav  --initl 0.01 --nepochs $nepochs ;























# datav=5
# dataname='waterdepth'
# #SET=$(seq 1 2000)
# SET=$(seq 1 20)

# for i in $SET
# do
#     python3 generate_real_waterdepth.py --num_saved $i --datav=$datav;               
# done










# datav=5
# dataname='waterdepth'
# python3 generate_real_waterdepth.py --num_totalsaved 3000 --datav=$datav;               









# datav=5
# dataname='waterdepth'
# nepochs=3000


# # #for l_scale in "1.0"; do
# # #for l_scale in "0.05" ".5"; do
# # #for l_scale in "1.0" ".5"; do
# # #for l_scale in "1.0" "0.1"; do
# # # for l_scale in "0.1" "0.01" "0.2"; do



# python3 test_realdataset_metatask.py  --modelname base    --datav $datav --dataname $dataname   --initl  0.1 --nepochs $nepochs ;
# python3 test_realdataset_metatask.py  --modelname base    --datav $datav --dataname $dataname   --initl  0.01 --nepochs $nepochs ;

#python3 test_realdataset_metatask.py  --modelname gpdep    --datav $datav --dataname $dataname   --initl 0.1 --nepochs $nepochs ;    
#python3 test_realdataset_metatask.py  --modelname gpdep2    --datav $datav --dataname $dataname   --initl 0.1 --nepochs $nepochs ;  #python3 test_realdataset_metatask.py  --modelname gpind    --datav $datav --dataname $dataname   --initl 0.1 --nepochs $nepochs ;    



















#l_scale=2.0



































#python3 test_training_correlatenp.py --tasktype mosm  --dep     --nepochs $nepochs --initl $l_scale --cnntype deep ; 


# python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl 0.01 --cnntype deep ; 
# python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype deep ; 

#python3 test_training_correlatenp.py --tasktype sin3          --nepochs $nepochs  --initl 1.0 --cnntype deep ; 
#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype deep ; 
#python3 test_training_correlatenp.py --tasktype sin3 --dep    --nepochs $nepochs --initl 0.1 --cnntype shallow ; 
