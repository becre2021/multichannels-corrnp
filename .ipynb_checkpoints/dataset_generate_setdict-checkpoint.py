import os
import time
import torch
import numpy as np
import torch.nn as nn 
import matplotlib.pyplot as plt
from dataset_multitask_1d import motask_generator

import argparse
parser = argparse.ArgumentParser(description='exp1')
parser.add_argument('--num_saved', type=int, default=0) 
parser.add_argument('--tasktype', type=str, default='lmc') #sin3, lmc, mogp
parser.add_argument('--testtype',type=str,default='inter') #inter extra

parser.add_argument('--dep', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')

#parser.add_argument('--dep',type=boolearn)

args = parser.parse_args()   



#-----------------------------------------
# globarl configuration
#-----------------------------------------
#tasktype = 'lmc'
tasktype = args.tasktype
nchannels = 3


def build_gen_cls(args):        
    testtype = args.testtype
    print('-'*100)
    print('build gen cls by tasktyp: {}, testtype: {}, dep: {}, train: {}'.format(args.tasktype,args.testtype,args.dep,args.train))
    print('-'*100)
    
    if args.testtype == 'inter':
        #train_range = [-4,4]
        #test_range = [-2,2]
        train_range = [0,10]
        test_range = [0,5]        
    elif args.testtype == 'extra':
        #train_range = [-2,2]
        #test_range = [-4,4]
        train_range = [0,3]
        test_range = [0,6]
        
    else:
        pass

    
    if args.tasktype == 'lmc':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
    elif args.tasktype == 'mosm':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
    elif args.tasktype =='sin3':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
    return gen_cls,train_range,test_range



#------------------
#build gen cls
#------------------
gen_cls,train_range,test_range = build_gen_cls(args)    


if args.tasktype in ['sin3']: 
    # intrain tr and validation
    ntask_train,nbatch_train=200,16
    ntask_train2,nbatch_train2= 50,16
    # intrain tr and validation
    ntask_test,nbatch_test=50,16
    ntask_test2,nbatch_test2= 50,16

if args.tasktype in ['lmc','mosm']:
    #---------------------
    # intrain tr and validation
    ntask_train,nbatch_train=100,12 #lmc
    ntask_train2,nbatch_train2= 25,12
    #---------------------
    # intrain tr and validation
    ntask_test,nbatch_test=25,12
    ntask_test2,nbatch_test2= 25,12





#if args.train == 'trainyes':
if args.train :
    save_path_set = './syndata_{}/dep{}_{}_{}.db'.format(tasktype,args.dep ,args.testtype, args.num_saved)
    #file_path = './syndata_lmc/depFalse_extra_78.db'    
    if not os.path.exists(save_path_set):

        print('generation train set ')


        nepochs = 1
        ntask,nbatch = ntask_train,nbatch_train
        ncontext,ntarget = np.random.randint(10,20),np.random.randint(20,40)        

        #for i in range(nepochs):
        context_x,context_y,target_x,target_y,full_x,full_y = [],[],[],[],[],[]
        #ncontext,ntarget = np.random.randint(10,50,2)        

        #---------------------------------------------------------------------------------------------------
        # training set in train
        #---------------------------------------------------------------------------------------------------        
        train_set ={}
        tic = time.time()
        for j in range(ntask):
            t_context_x,t_context_y,t_target_x,t_target_y,t_full_x,t_full_y = gen_cls.prepare_task(nbatch=nbatch,
                                                                                                   ncontext=ncontext,
                                                                                                   ntarget=ntarget,
                                                                                                   train_range=gen_cls.train_range,
                                                                                                   test_range=gen_cls.test_range,
                                                                                                   noise_true=True,
                                                                                                   intrain = True)
            #print(t_context_y.size(),t_target_y.size())

            context_x.append(t_context_x[None,:,:])
            context_y.append(t_context_y[None,:,:])
            target_x.append(t_target_x[None,:,:])
            target_y.append(t_target_y[None,:,:])
            full_x.append(t_full_x[None,:,:])
            full_y.append(t_full_y[None,:,:])


        context_x = torch.cat(context_x,dim=0)
        context_y = torch.cat(context_y,dim=0)
        target_x = torch.cat(target_x,dim=0)
        target_y = torch.cat(target_y,dim=0)
        full_x = torch.cat(full_x,dim=0)
        full_y = torch.cat(full_y,dim=0)



        #print(context_x.size())
        #print('')
        train_set = {'context_x':context_x,
                    'context_y':context_y,
                    'target_x':target_x,
                    'target_y':target_y,
                    'full_x':full_x,
                    'full_y':full_y}

        print('train taken {}'.format( np.round(time.time() - tic, 3) ))    
        print('train dict done')            
        print('intrain context {}, target {}'.format(context_x.shape,target_x.shape))
        
        del context_x
        del context_y
        del target_x
        del target_y
        del full_x
        del full_y

        torch.cuda.empty_cache()
        #torch.empty_cache()




        #---------------------------------------------------------------------------------------------------
        # valid set in train
        #---------------------------------------------------------------------------------------------------            
        #nepochs = 3
        #ntask = 2**5
        #nbatch = 2**4
        ntask, nbatch = ntask_train2,nbatch_train2
        ncontext,ntarget = 10,40     
        
        valid_set = {}
        #for i in range(nepochs):
        context_x,context_y,target_x,target_y,full_x,full_y = [],[],[],[],[],[]

        tic = time.time()
        for j in range(ntask):
            t_context_x,t_context_y,t_target_x,t_target_y,t_full_x,t_full_y = gen_cls.prepare_task(nbatch=nbatch,
                                                                                                   ncontext=ncontext,
                                                                                                   ntarget=ntarget,
                                                                                                   train_range=gen_cls.train_range,
                                                                                                   test_range=gen_cls.test_range,
                                                                                                   noise_true=True,
                                                                                                   intrain = True)
            #print(t_context_y.size(),t_target_y.size())

            context_x.append(t_context_x[None,:,:])
            context_y.append(t_context_y[None,:,:])
            target_x.append(t_target_x[None,:,:])
            target_y.append(t_target_y[None,:,:])
            full_x.append(t_full_x[None,:,:])
            full_y.append(t_full_y[None,:,:])


        context_x = torch.cat(context_x,dim=0)
        context_y = torch.cat(context_y,dim=0)
        target_x = torch.cat(target_x,dim=0)
        target_y = torch.cat(target_y,dim=0)
        full_x = torch.cat(full_x,dim=0)
        full_y = torch.cat(full_y,dim=0)


        valid_set = {'context_x':context_x,
                        'context_y':context_y,
                        'target_x':target_x,
                        'target_y':target_y,
                        'full_x':full_x,
                        'full_y':full_y}

        print('valid th taken {}'.format( np.round(time.time() - tic, 3) ))
        print('intrain context {}, target {}'.format(context_x.shape,target_x.shape))        
        print('valid dict done')    
        
        del context_x
        del context_y
        del target_x
        del target_y
        del full_x
        del full_y




        #db = {'train_set':train_set, 'valid_set':valid_set,'test_set':test_set}
        db = {'train_set':train_set, 'valid_set':valid_set}
        #save_path_set = './syndata_lmc/{}_{}_{}.db'.format(tasktype , testtype,args.num_saved)
        #save_path_set = './syndata_lmc/{}_{}_{}.db'.format(tasktype ,args.testtype, args.num_saved)
        save_path_set = './syndata_{}/dep{}_{}_{}.db'.format(tasktype,args.dep ,args.testtype, args.num_saved)

        torch.save(db, save_path_set)
        print('#'*100)
        print(save_path_set)
        print('\n'*5)    
    
    else:
        print('#'*100)
        print(save_path_set + '_already exist')
        print('\n'*5)    
        
    
    
    
    
    
    
    
    
#test mode gen    
else:
    print('generation test set ')
    
    
    nepochs = 1
    ntask, nbatch = ntask_test,nbatch_test
    ncontext,ntarget = 10,40        
    train_set1 ={}

    context_x,context_y,target_x,target_y,full_x,full_y = [],[],[],[],[],[]
    #---------------------------------------------------------------------------------------------------
    # train set in evaluation
    #---------------------------------------------------------------------------------------------------                
    tic = time.time()
    for j in range(ntask):
        t_context_x,t_context_y,t_target_x,t_target_y,t_full_x,t_full_y = gen_cls.prepare_task(nbatch=nbatch,
                                                                                               ncontext=ncontext,
                                                                                               ntarget=ntarget,
                                                                                               train_range=gen_cls.train_range,
                                                                                               test_range=gen_cls.test_range,
                                                                                               noise_true=True,
                                                                                               intrain = True) 
                                                                                               #intrain:True --> chose parameters """in""" training range
                                                                                               #intrain:false --> chose parameters """beyond""" training range
                    
        #print(t_context_y.size(),t_target_y.size())

        context_x.append(t_context_x[None,:,:])
        context_y.append(t_context_y[None,:,:])
        target_x.append(t_target_x[None,:,:])
        target_y.append(t_target_y[None,:,:])
        full_x.append(t_full_x[None,:,:])
        full_y.append(t_full_y[None,:,:])


    context_x = torch.cat(context_x,dim=0)
    context_y = torch.cat(context_y,dim=0)
    target_x = torch.cat(target_x,dim=0)
    target_y = torch.cat(target_y,dim=0)
    full_x = torch.cat(full_x,dim=0)
    full_y = torch.cat(full_y,dim=0)



    #print(context_x.size())
    #print('')
    train_set1 = {'context_x':context_x,
                'context_y':context_y,
                'target_x':target_x,
                'target_y':target_y,
                'full_x':full_x,
                'full_y':full_y}

    del context_x
    del context_y
    del target_x
    del target_y
    del full_x
    del full_y

    torch.cuda.empty_cache()
    #torch.empty_cache()

    print('train taken {}'.format( np.round(time.time() - tic, 3) ))    
    print('train dict done')    


    
    
    
    
    
    
    
    #---------------------------------------------------------------------------------------------------
    # test set in evaluation
    #---------------------------------------------------------------------------------------------------                    
    #nepochs = 3
    #ntask = 2**5
    #nbatch = 2**4
    valid_set1 = {}

    
    ntask, nbatch = ntask_test2,nbatch_test2
    ncontext,ntarget = 10,40         
    context_x,context_y,target_x,target_y,full_x,full_y = [],[],[],[],[],[]

    tic = time.time()
    for j in range(ntask):
        t_context_x,t_context_y,t_target_x,t_target_y,t_full_x,t_full_y = gen_cls.prepare_task(nbatch=nbatch,
                                                                                               ncontext=ncontext,
                                                                                               ntarget=ntarget,
                                                                                               train_range=gen_cls.train_range,
                                                                                               test_range=gen_cls.test_range,
                                                                                               noise_true=True,
                                                                                               intrain = False)
        #print(t_context_y.size(),t_target_y.size())

        context_x.append(t_context_x[None,:,:])
        context_y.append(t_context_y[None,:,:])
        target_x.append(t_target_x[None,:,:])
        target_y.append(t_target_y[None,:,:])
        full_x.append(t_full_x[None,:,:])
        full_y.append(t_full_y[None,:,:])


    context_x = torch.cat(context_x,dim=0)
    context_y = torch.cat(context_y,dim=0)
    target_x = torch.cat(target_x,dim=0)
    target_y = torch.cat(target_y,dim=0)
    full_x = torch.cat(full_x,dim=0)
    full_y = torch.cat(full_y,dim=0)

    valid_set1 = {'context_x':context_x,
                    'context_y':context_y,
                    'target_x':target_x,
                    'target_y':target_y,
                    'full_x':full_x,
                    'full_y':full_y}

    del context_x
    del context_y
    del target_x
    del target_y
    del full_x
    del full_y

    print('valid th taken {}'.format( np.round(time.time() - tic, 3) ))
    print('valid dict done')    

    
    #print('intest context {}, target {}'.format(context_x.shape,target_x.shape))
    
    
    #db = {'train_set':train_set, 'valid_set':valid_set,'test_set':test_set}
    db = {'train_set':train_set1, 'valid_set':valid_set1}
    #save_path_set = './syndata_lmc/{}_{}_{}.db'.format(tasktype ,args.testtype, -1)
    save_path_set = './syndata_{}/dep{}_{}_{}.db'.format(tasktype,args.dep ,args.testtype, args.num_saved)
    
    torch.save(db, save_path_set)
    print(save_path_set)
    print('#'*10)
    
    print('\n'*3)    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# #---------------------------------------------------------------------------------------------
# #test dict
# #---------------------------------------------------------------------------------------------
# nepochs = 1
# ntask = 2**2
# nbatch = 2**4
# test_set = {}
# for i in range(nepochs):
#     context_x,context_y,target_x,target_y,full_x,full_y = [],[],[],[],[],[]
#     ncontext,ntarget = np.random.randint(3,50,2)        
#     for j in range(ntask):
#         t_context_x,t_context_y,t_target_x,t_target_y,t_full_x,t_full_y = gen_cls.prepare_task(nbatch=nbatch,
#                                                                                                ncontext=ncontext,
#                                                                                                ntarget=ntarget,
#                                                                                                train_range=gen_cls.train_range,
#                                                                                                test_range=gen_cls.test_range,
#                                                                                                noise_true=False,
#                                                                                                intrain = False)
#         #print(t_context_y.size(),t_target_y.size())

#         context_x.append(t_context_x[None,:,:])
#         context_y.append(t_context_y[None,:,:])
#         target_x.append(t_target_x[None,:,:])
#         target_y.append(t_target_y[None,:,:])
#         full_x.append(t_full_x[None,:,:])
#         full_y.append(t_full_y[None,:,:])


#     context_x = torch.cat(context_x,dim=0)
#     context_y = torch.cat(context_y,dim=0)
#     target_x = torch.cat(target_x,dim=0)
#     target_y = torch.cat(target_y,dim=0)
#     full_x = torch.cat(full_x,dim=0)
#     full_y = torch.cat(full_y,dim=0)

#     #print('')
#     test_set[i] = {'context_x':context_x,
#                     'context_y':context_y,
#                     'target_x':target_x,
#                     'target_y':target_y,
#                     'full_x':full_x,
#                     'full_y':full_y}    
        

        
        
        
