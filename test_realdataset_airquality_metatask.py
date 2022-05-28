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
random_seed = 1111
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from collections import OrderedDict
import mogptk





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





df = pd.read_csv('data/AirQualityUCI.csv', delimiter=';') #03/10

# Replace missing values with NaN
df.replace(-200.0, np.nan, inplace=True)

df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S')
#
# start_stamp = '2005-03-01 00:00:00.0'
# mid_stamp = '2005-03-30 00:00:00.0'
# end_stamp = '2005-04-30 00:00:00.0'

start_stamp = '2005-03-1 00:00:00.0'
mid_stamp = '2005-03-20 00:00:00.0'
end_stamp = '2005-03-30 00:00:00.0'

#df['Date'] < pd.Timestamp(end_stamp) and 
ini_date = pd.Timestamp(start_stamp)
df1 = df[df['Date'] > pd.Timestamp(start_stamp)]
df2 = df1[df1['Date'] < pd.Timestamp(end_stamp)]
df2['n_time'] = (df2['Date'] - ini_date) / pd.Timedelta(hours=1)


# split train and target 
n_time_split = (pd.Timestamp(mid_stamp) - pd.Timestamp(start_stamp))/ pd.Timedelta(hours=1)

#cols = ['CO(GT)', 'T', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
cols = ['CO(GT)', 'T', 'NOx(GT)', 'NO2(GT)']

dataset = mogptk.LoadDataFrame(df2, x_col='n_time', y_col=cols)
num_channels = len(cols)

time_scale = 24
#normalize=False
normalize=True
if normalize == True:
    n_time_split /= time_scale

    
dataset_dict = OrderedDict()
X,Y = [],[]
for ith,ith_col in enumerate(cols):
    #dataset_dict[ith_col] = {}
    dataset_dict[ith] = {}
    
    x,y = dataset[ith_col].get_data()    
    #print(x)
    #dataset_dict[ith_col]['x'] = np.array(x,np.float32).squeeze()
    #dataset_dict[ith_col]['y'] = np.array(y,np.float32).squeeze()
    dataset_dict[ith]['x'] = np.array(x,np.float32).squeeze()
    dataset_dict[ith]['y'] = np.array(y,np.float32).squeeze()

    if normalize == True:
        dataset_dict[ith]['ymean'] = dataset_dict[ith]['y'].mean()
        dataset_dict[ith]['ystd'] = dataset_dict[ith]['y'].std() 
        dataset_dict[ith]['y'] = (dataset_dict[ith]['y'] - dataset_dict[ith]['ymean'])/dataset_dict[ith]['ystd']        
        
        dataset_dict[ith]['x'] /= time_scale 
        #n_time_split /= time_scale


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
def nparray2tensor(context_x,context_y,target_x,target_y):
    return torch.tensor(context_x).float(),           torch.tensor(context_y).float(),           torch.tensor(target_x).float(),           torch.tensor(target_y).float()

def get_index_new(dataset_dict,n_time_split):
    time_dict_index = {}    
    for ith_channel in dataset_dict:
        train_idx = np.where( (dataset_dict[ith_channel]['x'] <= n_time_split))[0]
        test_idx = np.where( ~ (dataset_dict[ith_channel]['x'] <= n_time_split))[0]
        full_idx = np.arange(len(dataset_dict[ith_channel]['x']))
        time_dict_index[ith_channel] = {'train_idx':train_idx,'test_idx':test_idx,'full_idx':full_idx}    
    return time_dict_index




def prepare_batch_new(dataset_dict, time_dict_index , nbatch = 32,batch_npoints=(64,64), intrain = True, forfig = False):
    context_x,context_y = [],[]
    target_x,target_y = [],[]
    full_x,full_y = [],[]
    
    #n_points = len(x)
    #time_dict_index = get_index(dataset_dict,test_option = test_option,test_len=test_len)
    
    for _ in range(nbatch):

        i_context_x,i_context_y = [],[]
        i_target_x,i_target_y = [],[]
        i_full_x,i_full_y = [],[]
        for ith_channel in time_dict_index:
            if intrain and not forfig:
                sampled_c_idx = np.random.choice(time_dict_index[ith_channel]['train_idx'],min(batch_npoints[0],len(time_dict_index[ith_channel]['train_idx'])))
                sampled_t_idx = np.random.choice(time_dict_index[ith_channel]['train_idx'],min(batch_npoints[1],len(time_dict_index[ith_channel]['train_idx'])))
                

            if not intrain and not forfig:                
                sampled_c_idx = np.random.choice(time_dict_index[ith_channel]['test_idx'],min(batch_npoints[0],len(time_dict_index[ith_channel]['test_idx'])))
                sampled_t_idx = np.random.choice(time_dict_index[ith_channel]['test_idx'],min(batch_npoints[1],len(time_dict_index[ith_channel]['test_idx'])))
                #ith_context_x,ith_context_y = dataset_dict[ith_channel]['x'][sampled_c_idx],dataset_dict[ith_channel]['y'][sampled_c_idx]
                #ith_target_x,ith_target_y = dataset_dict[ith_channel]['x'][sampled_t_idx],dataset_dict[ith_channel]['y'][sampled_t_idx]        
                
                
            if not intrain and forfig:                
                sampled_c_idx = np.random.choice(time_dict_index[ith_channel]['full_idx'],min(batch_npoints[0],len(time_dict_index[ith_channel]['full_idx'])))
                sampled_t_idx = np.random.choice(time_dict_index[ith_channel]['full_idx'],min(batch_npoints[1],len(time_dict_index[ith_channel]['full_idx'])))
                
                                
                
            sampled_c_idx.sort() 
            sampled_t_idx.sort()                 
            ith_context_x,ith_context_y = dataset_dict[ith_channel]['x'][sampled_c_idx],dataset_dict[ith_channel]['y'][sampled_c_idx]
            ith_target_x,ith_target_y = dataset_dict[ith_channel]['x'][sampled_t_idx],dataset_dict[ith_channel]['y'][sampled_t_idx]        

        
                #print('ith_target_x,ith_target_y ')
                #print(ith_target_x,ith_target_y )

            i_context_x.append(ith_context_x)
            i_context_y.append(ith_context_y) 
            i_target_x.append(ith_target_x)
            i_target_y.append(ith_target_y) 
            #i_full_x.append(dataset_dict[ith_channel]['x'])
            #i_full_y.append(dataset_dict[ith_channel]['y']) 



        i_context_x,i_context_y = np.asarray(i_context_x).T,np.asarray(i_context_y).T
        i_target_x,i_target_y = np.asarray(i_target_x).T,np.asarray(i_target_y).T   
        i_full_x,i_full_y = np.asarray(i_full_x).T,np.asarray(i_full_y).T   

        #print(i_full_x.shape)
        
        #print(i_context_x.shape,i_context_y.shape,i_target_x.shape,i_target_y.shape)
        
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




#-----------------------
# set time dict
time_dict_index = get_index_new(dataset_dict,n_time_split)
for j in range(num_channels):
    print(len(time_dict_index[j]['train_idx']),len(time_dict_index[j]['test_idx']),len(time_dict_index[j]['train_idx'])+len(time_dict_index[j]['test_idx']))



    
    
    









#-------------------------------------------
# train
#------------------------------------------
from test_baseline import Convcnp,compute_loss_baseline
from test_baseline_latent import  Convcnp_latent, compute_loss_baselinelatent
from test_dep_correlatenp import  DCGP_Convnp,compute_loss_gp  
from test_ind_correlatenp import  ICGP_Convnp,compute_loss_gp  


import argparse
parser = argparse.ArgumentParser(description='exp2')
parser.add_argument('--modelname',type=str,default= 'gpind')
parser.add_argument('--initl', type=float, default = 0.1)
parser.add_argument('--nepochs',type=int,default =5000)

# parser.add_argument('--lr', type=float, default= 0.001) #iterations
# parser.add_argument('--weightdecay', type=float, default= 1e-4)

args = parser.parse_args()   
expinfo = make_expinfo(args)
print('#'+'-'*100)
print(expinfo[:-1])
print('#'+'-'*100)
print('\n')






#num_channels = 5
#num_channels = num_channels = len(cols)
nsamples_latent=10
nsamples_gp = 10
cnntype = 'shallow'
lr = 0.001
weight_decay=1e-4
#cnntype = 


init_lengthscale = args.initl

#def get_model(modelname='gp'):        
def get_model(modelname='gpdep',cnntype='deep'):        
    
    if modelname == 'base':
        model = Convcnp(in_dims=1,out_dims=1,num_channels=num_channels,cnntype=cnntype,init_lengthscale=init_lengthscale).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_baseline

    if modelname == 'baselatent':
        model = Convcnp_latent(in_dims=1,out_dims=1,num_channels=num_channels,num_postsamples=nsamples_latent,cnntype=cnntype,init_lengthscale=init_lengthscale).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_baselinelatent
        
    #if modelname == 'gp_ind':
    if modelname == 'gpind':        
        model = ICGP_Convnp(in_dims=1,out_dims=1,num_channels=num_channels,num_postsamples=nsamples_gp,cnntype=cnntype,init_lengthscale=init_lengthscale).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_gp
        
        
    #if modelname == 'gp_dep':
    if modelname == 'gpdep':        
        model = DCGP_Convnp(in_dims=1,out_dims=1,num_channels=num_channels,num_postsamples=nsamples_gp,cnntype=cnntype,init_lengthscale=init_lengthscale).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_gp
        
      
    return model,opt,lossfun




def merge_allset_1d(xc,yc,xt,yt):
    xct = torch.cat([xc,xt],dim=1)
    yct = torch.cat([yc,yt],dim=1)
    xct,s_idx =torch.sort(xct,dim=1)

    if len(xc.size()) == 3:
        yct = torch.gather(yct,1,s_idx)    
    if len(xc.size()) == 4:
        yct = torch.gather(yct,1,s_idx[:,:,0,:])
    return xct,yct
    
    
#reglamda=1.
proposed_model_list = ['gpind','gpdep']
#def train_epochs_with_dict(set_dict_epoch,model,opt,lossfun,trainmodel='convcnp',trainprogress_ratio = 0.0):
def train_epochs(dataset_dict,time_dict_index,model,opt,lossfun,ntask=4,nbatch=32,ncontext=32,ntarget=2*32):
    
    model.train()
    likelihoods = []
    
    #ntask = set_dict_epoch['context_x'].size(0)
    for _ in range(ntask):
                
        context_x,context_y,target_x,target_y = prepare_batch_new(dataset_dict, 
                                                                  time_dict_index , 
                                                                  nbatch = nbatch,
                                                                  batch_npoints=(ncontext ,ntarget), 
                                                                  intrain = True)        
        
        #context_x,context_y,target_x,target_y = nparray2tensor(context_x,context_y,target_x,target_y)        
        if model.modelname in proposed_model_list and len(context_x.size()) == 3:        
            context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)        
        
        target_x,target_y = merge_allset_1d(context_x,context_y,target_x,target_y)

        #predict & train models
        y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())    
        obj = lossfun( y_mean,y_std, target_y.cuda())

        obj.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        opt.step()
        opt.zero_grad()
        
        #losses.append(obj.item())
        likelihoods.append(obj.cpu().data.numpy())        
        
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       




#def validate_epochs(dataset_dict,time_dict_index,model,ntask=128,nbatch=32,ncontext=32,ntarget=2*32,train_range= None,test_range=None,intrain=True):
def validate_epochs(dataset_dict,time_dict_index,model,lossfun,ntask=128,nbatch=4,ncontext=32,ntarget=2*32,train_range= None,test_range=None,intrain=True):

    # large is better 
    model.eval()
    likelihoods = []
    
    for _ in range(ntask):        
        
        context_x,context_y,target_x,target_y = prepare_batch_new(dataset_dict, 
                                                                  time_dict_index , 
                                                                  nbatch = nbatch,
                                                                  batch_npoints=(ncontext ,ntarget), 
                                                                  intrain = intrain)

         
        if model.modelname in proposed_model_list and len(context_x.size()) == 3:                
            context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)
                    
        
        
        y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())        
        #obj = -compute_nll( y_mean,y_std, target_y.cuda())
        obj = -lossfun( y_mean,y_std, target_y.cuda())
        likelihoods.append(obj.cpu().data.numpy())        
                
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       
    


    
    
    
    
    
    
    







#----------------------------------------------
#tarining config
#----------------------------------------------

#nepochs=3000
#nepochs=6000
#nepochs=200


modelname = args.modelname
saved_modelparam_path = './param_airquality/{}_{}_nepochs{}_initl{}.pth'.format(modelname,cnntype,args.nepochs,args.initl)
model,opt,lossfun = get_model(modelname=modelname,cnntype=cnntype)
#model.num_params

#-----------------------------
#wandbi name assignment
#-----------------------------    
config = {'tasktype':'airquality',
          'model':model.modelname,
          'initl':args.initl,
          'nepochs':args.nepochs,
          'cnntype':cnntype}

wandb.init( project="uai22-9234-airquality",
            config = config,
            reinit= True)

wandb.define_metric("tmp")    

#wandb.run.name = '{}_{}_nepochs{}_initl{}'.format(model.modelname,cnntype,args.nepochs,args.initl)    
wandb.run.name = '{}_{}_nepochs{}_initl{}_v2'.format(model.modelname,cnntype,args.nepochs,args.initl)    

wandb.run.save()

# watch model
wandb.watch(model)


# nbatch=4
best_loss = -np.inf
nepochs = args.nepochs
#ntask,nbatch,ncontext,ntarget = 1,4,20,100
ntask,nbatch = 1,8

for i in range(1,nepochs + 1):   
    #ncontext_r = np.random.randint(10,50)  

    epoch_start = time.time()
    ncontext_r = np.random.randint(10,30)      
    ntarget_r =np.random.randint(50,100)  
    avg_loss,loss_list = train_epochs(dataset_dict,time_dict_index,model,opt,lossfun,
                                      ntask=ntask,nbatch=nbatch,ncontext=ncontext_r,ntarget=ntarget_r)


    #ncontext,ntarget = np.random.randint(10,50,2)        
    ncontext,ntarget = 25,100        
    val_loss,val_loss_std = validate_epochs(dataset_dict,time_dict_index,model,lossfun,
                                             ntask=ntask,nbatch=nbatch,ncontext=ncontext,ntarget=ntarget)



    epoch_end = time.time()
    if best_loss < val_loss:
        best_loss = val_loss        
        saved_dict = {'epoch': i ,
                     'best_acc_top1': best_loss,                         
                     'state_dict': model.state_dict(),
                     'optimizer': opt.state_dict()}
        torch.save(saved_dict,saved_modelparam_path)
        saved_epoch = i
        #print('epochs [{}/{}] | param_saved at {}'.format(i,nepochs,saved_modelparam_path ))

    if i%50 ==0 or i == 1 :
        print('epochs [{}/{}] | train loss {:.3f}, val loss {:.3f}, best val loss {:.3f} \t with epoch {} \t taken time {:.2f}'.format(i,nepochs,avg_loss,val_loss,best_loss,saved_epoch,epoch_end-epoch_start))       
        #print(loss_list)    

        #wandbi tarinining check
        wandb.log({"trnll-intrain": avg_loss,"valnll-intrain": val_loss,'current_epoch':i})



    torch.cuda.empty_cache()

print('training done')
print('\n'*5)




path1 = saved_modelparam_path 
model,_,lossfun = get_model(modelname=modelname,cnntype=cnntype)
load_dict = torch.load(path1)
model.load_state_dict(load_dict['state_dict'])   

modelspec = '{}_{}_nsample{}_initl{}.pth'.format(modelname,cnntype,nsamples_gp,init_lengthscale )
ntask,nbatch,_,ntarget = 64,4,_,200
ncontext_list = [10,20,30,50]
results_list = []
for ncontext in ncontext_list:
    #val_loss_m,val_loss_s = validate_epochs(dataset_dict,time_dict_index,model,ntask=ntask,nbatch=nbatch,ncontext=ncontext,ntarget=ntarget, intrain=False)
    val_loss_m,val_loss_s = validate_epochs(dataset_dict,time_dict_index,model,lossfun,
                                            ntask=ntask,nbatch=nbatch,ncontext=ncontext,ntarget=ntarget , intrain=False)
    results_list.append((val_loss_m,val_loss_s))
    log_dict = {"outtest_ncontext":ncontext,
                "outtest_nllmean":val_loss_m,
                "outtest_nllstd":val_loss_s,
                "saved_epoch":saved_epoch}                
    wandb.log(log_dict)
