#!/usr/bin/env python
# coding: utf-8


import wandb
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 
import random


# random_seed = 1111
# torch.manual_seed(random_seed)
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from collections import OrderedDict
import mogptk




import argparse
parser = argparse.ArgumentParser(description='exp1')
#parser.add_argument('--num_saved', type=int, default=0) 
parser.add_argument('--num_totalsaved', type=int, default=200) 

parser.add_argument('--datav', type=int, default=2) 


# parser.add_argument('--tasktype', type=str, default='lmc') #sin3, lmc, mogp

#parser.add_argument('--dep',type=boolearn)
args = parser.parse_args()   








device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()


def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    for ith_keys in args_dict:
        if ith_keys in ['csvpath']:
            pass
        else:
            model_info += ith_keys + str(args_dict[ith_keys]) + '_'
    return model_info




from collections import OrderedDict


cols = ['bramblemet','cambermet','chimet','sotonmet']
start_date = '2020-05-01'
starts =  pd.to_datetime('01/05/2020 00:00', format='%d/%m/%Y %H:%M')

end_date = '2020-07-01'
#end_date_ = '2020-06-30'
#totaldays=22+30


#totalnumobs = totaldays*24*12
print('Loading Bramble')
bramblemet = pd.read_csv('data/bramblemet/bramblemet.csv.gz', compression='gzip', low_memory=False)
bramblemet['Date'] = pd.to_datetime(bramblemet['Date'] + ' ' + bramblemet['Time'], format='%d/%m/%Y %H:%M')
bramblemet = bramblemet.drop(columns=['Time'])
bramblemet = bramblemet.loc[(bramblemet['Date'] >= start_date) & (bramblemet['Date'] < end_date)]

print('Loading Camber')
cambermet = pd.read_csv('data/bramblemet/cambermet.csv.gz', compression='gzip', low_memory=False)
cambermet['Date'] = pd.to_datetime(cambermet['Date'] + ' ' + cambermet['Time'], format='%d/%m/%Y %H:%M')
cambermet = cambermet.drop(columns=['Time'])
cambermet = cambermet.loc[(cambermet['Date'] >= start_date) & (cambermet['Date'] < end_date)]

print('Loading Chi')
chimet = pd.read_csv('data/bramblemet/chimet.csv.gz', compression='gzip', low_memory=False)
chimet['Date'] = pd.to_datetime(chimet['Date'] + ' ' + chimet['Time'], format='%d/%m/%Y %H:%M')
chimet = chimet.drop(columns=['Time'])
chimet = chimet.loc[(chimet['Date'] >= start_date) & (chimet['Date'] < end_date)]

print('Loading Soton')
sotonmet = pd.read_csv('data/bramblemet/sotonmet.csv.gz', compression='gzip', low_memory=False)
sotonmet['Date'] = pd.to_datetime(sotonmet['Date'] + ' ' + sotonmet['Time'], format='%d/%m/%Y %H:%M')
sotonmet = sotonmet.drop(columns=['Time'])
sotonmet = sotonmet.loc[(sotonmet['Date'] >= start_date) & (sotonmet['Date'] < end_date)]


# startfloat = sotonmet['Date'][sotonmet['Date'] == '2020-06-01 00:00:00'].to_numpy().astype('float32') 
# endfloat = sotonmet['Date'][sotonmet['Date'] == '2020-06-30 23:55:00'].to_numpy().astype('float32')

# startfloat = chimet['Date'][chimet['Date'] == start_date + ' 00:00:00'].to_numpy().astype('float32') 
# endfloat = sotonmet['Date'][sotonmet['Date'] == end_date_ + ' 23:55:00'].to_numpy().astype('float32')

#---------------------------------------
# procesing dataset 
#---------------------------------------
normalize=True
dataset_dict = OrderedDict()
dataset_dict[0] = bramblemet
dataset_dict[1] = cambermet
dataset_dict[2] = chimet
dataset_dict[3] = sotonmet




dataset_dict['total_day'] = 61
dataset_dict['split_day'] = 31
dataset_dict['duration'] = 4
dataset_dict['unit'] = 24
dataset_dict['unit2'] = 1


num_channels = 4
default_round = 10**11

for ith_dict in range(num_channels):    
    #x = dataset_dict[ith_dict]['Date'].to_numpy().astype('int')/default_round
    #x = dataset_dict[ith_dict]['Date'].to_numpy(dtype=np.float32)     
    #x = ((dataset_dict[ith_dict]['Date']  -starts)  / (pd.Timedelta('1s')*60*60)).to_numpy(dtype=np.float32)           
    #x = ((dataset_dict[ith_dict]['Date']  -starts)  / (24*(pd.Timedelta('1h'))) ).to_numpy(dtype=np.float32)           
    x = ((dataset_dict[ith_dict]['Date']  -starts)  / (1*(pd.Timedelta('1h'))) ).to_numpy(dtype=np.float32)           
    
    y = dataset_dict[ith_dict]['DEPTH'].to_numpy(dtype=np.float32)    
    #dataset_dict[ith_dict] = 
    #x,y = np.asarray(x,dtype=np.float32),np.asarray(y,dtype=np.float32)
    
    dataset_dict[ith_dict] = {}
    dataset_dict[ith_dict]['x'] = x
    dataset_dict[ith_dict]['y'] = y
    
    
    x0 = dataset_dict[ith_dict]['x'][0]
    if normalize == True:
        #dataset_dict[ith_dict]['x'] -= x0 
        #dataset_dict[ith_dict]['x'] /= time_scale        
        dataset_dict[ith_dict]['ymean'] = dataset_dict[ith_dict]['y'].mean()
        dataset_dict[ith_dict]['ystd'] = dataset_dict[ith_dict]['y'].std() 
        dataset_dict[ith_dict]['y'] = (dataset_dict[ith_dict]['y'] - dataset_dict[ith_dict]['ymean'])/dataset_dict[ith_dict]['ystd']

    idx = np.argsort(dataset_dict[ith_dict]['x'])
    dataset_dict[ith_dict]['x'] = dataset_dict[ith_dict]['x'][idx]/dataset_dict['unit']
    dataset_dict[ith_dict]['y'] = dataset_dict[ith_dict]['y'][idx]        

    #rescale_x
    dataset_dict[ith_dict]['x'] = dataset_dict[ith_dict]['x'][::12]
    dataset_dict[ith_dict]['y'] = dataset_dict[ith_dict]['y'][::12]
    



        
        
        
        
        
        
        
        
        
        
#-----------------------
# define utility function        
#-----------------------
def nparray2tensor(context_x,context_y,target_x,target_y):
    return torch.tensor(context_x).float(),\
           torch.tensor(context_y).float(),\
           torch.tensor(target_x).float(),\
           torch.tensor(target_y).float()


def prepare_batch_new(dataset_dict , nbatch = 32,batch_npoints=(64,64), intrain = True, forfig = False,num_channels=4):
    context_x,context_y = [],[]
    target_x,target_y = [],[]
    full_x,full_y = [],[]
    
    #n_points = len(x)
    #time_dict_index = get_index(dataset_dict,test_option = test_option,test_len=test_len)
    split_day = dataset_dict['split_day']
    for _ in range(nbatch):

        i_context_x,i_context_y = [],[]
        i_target_x,i_target_y = [],[]
        i_full_x,i_full_y = [],[]
        #for ith_channel in timedict_index:
        
        if intrain and not forfig:
            chosen_day = np.random.randint(5,dataset_dict['split_day']-dataset_dict['duration'])
        if not intrain and not forfig:                
            chosen_day = np.random.randint(dataset_dict['split_day']+dataset_dict['duration'],dataset_dict['total_day']-dataset_dict['duration'])                                  
        if not intrain and forfig:    
            chosen_day = np.random.randint(dataset_dict['split_day']+dataset_dict['duration'],dataset_dict['total_day']-dataset_dict['duration'])        
        #print(dayindex)
                
        #for ith_channel in dataset_dict.keys():    
        for ith_channel in range(num_channels):    
            
            if not forfig:                
                #v1
                #index_candidate = np.where(   (chosenday-1<dataset_dict[ith_channel]['x'])  & (dataset_dict[ith_channel]['x']<(chosenday+1)  ))[0]
                #v2
                #index_candidate = np.where(   (chosenday-3.5<dataset_dict[ith_channel]['x'])  & 
                #                              (dataset_dict[ith_channel]['x']<(chosenday + 3.5)  ))[0]
           
                lb,ub = dataset_dict['unit2'] *(chosen_day-dataset_dict['duration']),   dataset_dict['unit2'] *(chosen_day + dataset_dict['duration'])        
                index_candidate = np.where(  (lb < dataset_dict[ith_channel]['x'])  &  (dataset_dict[ith_channel]['x']< ub  ))[0] 

                assert len(index_candidate) >= batch_npoints[0]+batch_npoints[1]
                
            else:
                #index_candidate = np.arange(len(dataset_dict[ith_channel]['x']))

                #v1: full region                
                #index_candidate = np.where(   dataset_dict[ith_channel]['x'] >= split_day  )[0]

                #v2: test region                
                index_candidate = np.where(   (dataset_dict[ith_channel]['x'] >= split_day + 4*1))[0]

#                 #v3: test region                                
#                 index_candidate = np.where(   (chosenday-3.5<dataset_dict[ith_channel]['x'])  &
#                                                (dataset_dict[ith_channel]['x']<(chosenday + 3.5)  ))[0]
                 
                lb,ub = dataset_dict['unit2'] *(chosen_day-dataset_dict['duration']),   dataset_dict['unit2']*(chosen_day + dataset_dict['duration'])        
                index_candidate = np.where(  (lb < dataset_dict[ith_channel]['x'])  &  (dataset_dict[ith_channel]['x']< ub  ))[0] 

                
            #set index    
            sampled_c_idx = np.sort(np.random.choice(index_candidate,batch_npoints[0],replace=False))
            left_c_idx = np.setdiff1d(index_candidate,sampled_c_idx)
            sampled_t_idx = np.sort(np.random.choice(left_c_idx,  min(batch_npoints[1],len(left_c_idx)),replace=False ))
                
            #get context and target set     
            ith_context_x,ith_context_y = dataset_dict[ith_channel]['x'][sampled_c_idx],dataset_dict[ith_channel]['y'][sampled_c_idx]
            ith_target_x,ith_target_y = dataset_dict[ith_channel]['x'][sampled_t_idx],dataset_dict[ith_channel]['y'][sampled_t_idx]        


            i_context_x.append(ith_context_x)
            i_context_y.append(ith_context_y) 
            i_target_x.append(ith_target_x)
            i_target_y.append(ith_target_y) 



        i_context_x,i_context_y = np.asarray(i_context_x).T,np.asarray(i_context_y).T
        i_target_x,i_target_y = np.asarray(i_target_x).T,np.asarray(i_target_y).T   
        i_full_x,i_full_y = np.asarray(i_full_x).T,np.asarray(i_full_y).T   

        
        context_x.append( i_context_x )
        context_y.append( i_context_y )
        target_x.append( i_target_x  )
        target_y.append( i_target_y )
        #full_x.append(i_full_x)
        #full_y.append(i_full_y)
    
    #print(type(context_x))        

    context_x = np.asarray(context_x,dtype=np.float32)
    context_y = np.asarray(context_y,dtype=np.float32)
    target_x = np.asarray(target_x,dtype=np.float32)
    target_y = np.asarray(target_y,dtype=np.float32)
    full_x = np.asarray(full_x,dtype=np.float32)
    full_y = np.asarray(full_y,dtype=np.float32)
        
    context_x,context_y,target_x,target_y = nparray2tensor(context_x,context_y,target_x,target_y)
    return context_x,context_y,target_x,target_y

        
        
        
        
        





# ntask=128
#ntask = 32


#nepochs=args.nepochs
#best_loss = -np.inf
#for i in range(1,nepochs + 1):   

#num_totalsaved=100
nbatch = 16
#for num_saved in range(1,args.num_totalsaved+1): 

start0=1
for num_saved in range(start0,args.num_totalsaved+start0): 
    
    epoch_start = time.time()


    ncontext_r = np.random.randint(5,30)      
    ntarget_r = np.random.randint(30,50)     
    tr_context_x,tr_context_y,tr_target_x,tr_target_y = prepare_batch_new(dataset_dict, 
                                                                          nbatch = nbatch,
                                                                          batch_npoints=(ncontext_r ,ntarget_r), 
                                                                          intrain = True)            

    # print('tr_context_x,tr_context_y,tr_target_x,tr_target_y')
    # print(tr_context_x,tr_context_y,tr_target_x,tr_target_y)


    #ncontext,ntarget = np.random.randint(10,50,2)        
    #ncontext,ntarget = 10,40        
    ncontext,ntarget = 20,40        
    val_context_x,val_context_y,val_target_x,val_target_y = prepare_batch_new(dataset_dict, 
                                                                              nbatch = nbatch,
                                                                              batch_npoints=(ncontext ,ntarget), 
                                                                              intrain = True)


    # print('val_context_x,val_context_y,val_target_x,val_target_y')
    # print(val_context_x,val_context_y,val_target_x,val_target_y)



    epoch_end = time.time()


    tr_set = (tr_context_x,tr_context_y,tr_target_x,tr_target_y)
    val_set = (val_context_x,val_context_y,val_target_x,val_target_y)


    #db = {'train_set':train_set, 'valid_set':valid_set,'test_set':test_set}
    db = {'tr_set':tr_set, 'val_set':val_set}
    #save_path_set = './realdata_airquality/set_{}.db'.format(args.num_saved)
    #save_path_set = './realdata_airquality_2/set_{}.db'.format(args.num_saved)
    #save_path_set = './realdata_waterdepth/set_{}.db'.format(args.num_saved)
    save_path_set = './realdata_waterdepth_v{}/set_{}.db'.format(args.datav,num_saved )


    torch.save(db, save_path_set)    
    print('-'*100)    
    print('tr_context_x.shape,tr_context_y.shape,tr_target_x.shape,tr_target_y.shape')
    print(tr_context_x.shape,tr_context_y.shape,tr_target_x.shape,tr_target_y.shape)    
    print('val_context_x.shape,val_context_y.shape,val_target_x.shape,val_target_y.shape')
    print(val_context_x.shape,val_context_y.shape,val_target_x.shape,val_target_y.shape)    
    print('\n'*1)
    print('dataset :' + save_path_set + '_taken time(sec): {:.2f}'.format(epoch_end-epoch_start))
    print('-'*100)    
    print('\n'*2)

    print('done')








# path1 = saved_modelparam_path 
# model,_,lossfun = get_model(modelname=modelname,cnntype=cnntype)
# load_dict = torch.load(path1)
# model.load_state_dict(load_dict['state_dict'])   

# modelspec = '{}_{}_nsample{}_initl{}.pth'.format(modelname,cnntype,nsamples_gp,init_lengthscale )
# ntask,nbatch,_,ntarget = 64,4,_,20
# ncontext_list = [3,6,9,12]
# results_list = []
# for ncontext in ncontext_list:
#     #val_loss_m,val_loss_s = validate_epochs(dataset_dict,time_dict_index,model,ntask=ntask,nbatch=nbatch,ncontext=ncontext,ntarget=ntarget, intrain=False)
#     val_loss_m,val_loss_s = validate_epochs(dataset_dict,model,lossfun,
#                                             ntask=ntask,nbatch=nbatch,ncontext=ncontext,ntarget=ntarget , intrain=False)
#     results_list.append((val_loss_m,val_loss_s))
#     log_dict = {"outtest_ncontext":ncontext,
#                 "outtest_nllmean":val_loss_m,
#                 "outtest_nllstd":val_loss_s,
#                 "saved_epoch":saved_epoch}                
#     wandb.log(log_dict)
