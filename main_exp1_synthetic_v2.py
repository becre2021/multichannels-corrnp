import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
from dataset_multi import motask_generator
import numpy as np


# from model_proposed import ConvCNP_Multicross
# from model_proposed_v2 import G_ConvCNP_Multi
# from model_proposed_v3 import Cross_ConvCNP
# from model_proposed_v4 import Cross_ConvCNP_G,Cross_ConvCNP_L

from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet, UNet_Depth3
from convcnp.cnp_multi import RegressionANP, RegressionCNP

from model_baseline import ConvCNP_Multi
from model_latent_baseline import ConvCNP_Latent_Multi
from model_proposed_new_v1 import ConvCNP_Multi_CC
from model_proposed_new_v2 import ConvCNP_Multi_CC as ConvCNP_Multi_CCC

from train_loss import compute_nll,compute_nll_latent    



import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import argparse


import time 
import random
random_seed = 1111
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



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



parser = argparse.ArgumentParser(description='exp1-synthetics')
parser.add_argument('--tasktype', type=str, default='sin3') # sin3,sin4,mogp,lmc,convk,
parser.add_argument('--testtype', type=str, default='extra') # inter,extra
parser.add_argument('--nepochs', type=int, default=20) #iterations
parser.add_argument('--nchannels', type=int, default=3)
parser.add_argument('--dep', action='store_true')

args = parser.parse_args()   
expinfo = make_expinfo(args)
print('#'+'-'*100)
print(expinfo[:-1])
print('#'+'-'*100)
print('\n')








#--------------------------------
#tarining config
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nchannels = 3
in_channels = nchannels


iscuda = True
lr = 5e-4
weight_decay = 1e-4
points_per_unit = 64
nbasis = 5
init_l = 0.01


base_model_list = ['convcnp','convcnp_d6']    
latent_model_list = ['convcnp_latent','convcnp_d6_latent']    

def build_model(model_type):
    if model_type == 'convcnp':
        model = ConvCNP_Multi(in_channels=in_channels,
                              rho = UNet_Depth3(in_channels**2),
                              points_per_unit=points_per_unit,
                              rbf_init_l= init_l,
                              nbasis = nbasis)
    
    if model_type == 'convcnp_d6':
        model = ConvCNP_Multi(in_channels=in_channels,
                              rho = UNet(in_channels**2),
                              points_per_unit=points_per_unit,
                              rbf_init_l= init_l,
                              nbasis = nbasis)

    
    if model_type == 'convcnp_latent':
        model = ConvCNP_Latent_Multi( in_channels=in_channels,
                                      rho = UNet_Depth3(in_channels**2),
                                      points_per_unit=points_per_unit,
                                      rbf_init_l= init_l,
                                      nbasis = nbasis)

    if model_type == 'convcnp_d6_latent':
        model = ConvCNP_Latent_Multi( in_channels=in_channels,
                                      rho = UNet(in_channels**2),
                                      points_per_unit=points_per_unit,
                                      rbf_init_l= init_l,
                                      nbasis = nbasis)


    elif model_type == 'pro-nv2':        
        model = ConvCNP_Multi_CC(in_channels=in_channels,
                               rho = UNet_Depth3(in_channels**2),
                               cc_rho = UNet_Depth3(in_channels**2),
                               points_per_unit=points_per_unit,
                               rbf_init_l= init_l,
                               nbasis = nbasis)

    elif model_type == 'pro-nv22':        
        model = ConvCNP_Multi_CCC(in_channels=in_channels,
                               rho = UNet_Depth3(in_channels**2),
                               cc_rho = UNet_Depth3(in_channels**2),
                               points_per_unit=points_per_unit,
                               rbf_init_l= init_l,
                               nbasis = nbasis)
        
    elif model_type == 'anp':
        model = RegressionANP(input_dim=nchannels,
                              latent_dim=128,
                              num_channels=nchannels)        
        
        

    elif model_type == 'cnp':
        model = RegressionCNP(input_dim=nchannels,
                              latent_dim=128,
                              num_channels=nchannels)
    else:
        pass
    

    if iscuda:
        model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
    
    return model,opt
    

    

    
def train_epochs_with_dict(set_dict_epoch,model,opt,trainmodel='convcnp'):
    model.train()
    likelihoods = []
    
    ntask = set_dict_epoch['context_x'].size(0)
    for ith in range(ntask):
                
        context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
        target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]

        
        # predict & train models
        y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())        
        if trainmodel in base_model_list:
            obj = compute_nll( y_mean,y_std, target_y.cuda())
                        
        if trainmodel in latent_model_list:
            #y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())
            obj = compute_nll_latent( y_mean, y_std, target_y.cuda())
    
    
        # Optimization
        obj.backward()
        opt.step()
        opt.zero_grad()
        
        #losses.append(obj.item())
        likelihoods.append(obj.cpu().data.numpy())        
        
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       




def validate_epochs_with_dict(set_dict_epoch,model,test_range=None,trainmodel='convcnp'):
    # large is better 
    model.eval()
    likelihoods = []
    
    ntask = set_dict_epoch['context_x'].size(0)    
    for ith in range(ntask):        
        
        context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
        target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]
        y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())

        if trainmodel in base_model_list:
            #y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())
            obj = -compute_nll( y_mean,y_std, target_y.cuda())
               
        if trainmodel in latent_model_list:
            obj = -compute_nll_latent( y_mean, y_std, target_y.cuda())
            
            
        likelihoods.append(obj.cpu().data.numpy())        
                
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       
    

    
    
    
    
    
    
    
    
    
    
    
    
#-------------------------------------------
# task setup    
#-------------------------------------------
    
#tasktype = 'sin3'
#tasktype = 'lmc'
#tasktype = 'mosm'
#dep = True
#testtype = 'extra'

tasktype = args.tasktype
dep = args.dep
testtype = args.testtype


if testtype == 'inter':
    train_range = [-4,4]
    test_range = [-2,2]
    gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=dep)

elif testtype == 'extra':
    #train_range,test_range = [-2,2],[-4,4]
    train_range,test_range = [0,5],[5,10]   
    gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=dep)
else:
    pass






#-------------------------------------------
# train
#------------------------------------------
#lr = 1e-3
lr = 5e-4
#nepochs = 200
#nepochs = 400
#nepochs = 50
nepochs= args.nepochs

    
    
    
    
#-------------------------------------------
# wandb
#-------------------------------------------
import wandb
#compare_models_list = [('propose_n_v1','uniform'),('convcnp','uniform')]
#compare_models_list = [('pro-nv2','uniform'),('convcnp','uniform')]
#compare_models_list = [('pro-nv2','uniform')]
#compare_models_list = [('convcnp','uniform')]


#compare_models_list = [('pro-nv2','uniform'),('convcnp','uniform')]
#compare_models_list = [('pro-nv2','uniform')]
#compare_models_list = [('pro-nv22','uniform')]
#compare_models_list = [('convcnp_latent','uniform')]


#compare_models_list = [('convcnp_latent','latent')]



#compare_models_list = ['convcnp','convcnp_latent']
compare_models_list = ['convcnp_d6','convcnp_d6_latent']



saved_param_format = '.pth'
for model_type in compare_models_list:
    print('-'*100)
    #model_type,trainloss = jth_pair[0],jth_pair[1]
    #model_type = jth_pair
    model,opt = build_model(model_type=model_type)   
    #model_saved_name = 'modeltype{}_trainloss{}_nepochs{}_lr{}'.format(model_type,trainloss,nepochs,lr)    
    model_saved_name = 'modeltype{}_nepochs{}_lr{}'.format(model_type,nepochs,lr)    
    
    #saved_modelparam_path = './param_sin3/' + model_saved_name + '.pth'                                                                                                        
    saved_modelparam_path = './param_{}/'.format(tasktype) + model_saved_name + saved_param_format                                                    
    print('param_save_path : {}'.format(model_saved_name))
    print('\n')
    
    
    #-----------------------------
    #wandbi name assignment
    #-----------------------------    
    config = {'lr':lr,
              'weight_decay':weight_decay,
              'points_per_unit':points_per_unit,
              'nbasis':nbasis,
              'init_l':init_l,
              'nepochs':nepochs,
              'tasktype':tasktype,
              'testtype':testtype}
    
    
    wandb.init( project="crossconvcnp_v3",
                config = config,
                reinit= True)
    # train history
    wandb.define_metric("tr_mnll_intrain",   summary="min")
    wandb.define_metric("val_mnll_intrain",  summary="max")    
    wandb.define_metric("val_mnll_mean_withintrain")
    wandb.define_metric("val_mnll_stde_withintrain")    
    wandb.define_metric("te_mnll_mean_beyondtrain")    
    wandb.define_metric("te_mnll_stde_beyondtrain")    
    
    wandb.run.name = model_saved_name 
    wandb.run.save()
    

    
    best_loss = -np.inf
    for i in range(1,nepochs + 1):   
        try:
            save_path_set = './syndata_{}/dep{}_{}_{}'.format(tasktype, dep, testtype, i)
            print(save_path_set)
            loaded = torch.load(save_path_set + '.db')
            train_set = loaded['train_set']
            valid_set = loaded['valid_set']
            print('success load at {}'.format(save_path_set ))        
        except:
            print('failed load at {}'.format(save_path_set ))            
            pass

#         save_path_set = './syndata_{}/dep{}_{}_{}'.format(tasktype, dep, testtype, i)
#         print(save_path_set)
#         loaded = torch.load(save_path_set + '.db')
#         train_set = loaded['train_set']
#         valid_set = loaded['valid_set']

        #print('[{}/{}] epochs | modelparam : {}'.format(i,nepochs,saved_modelparam_path))
        avg_loss,std_loss = train_epochs_with_dict( train_set,model,opt,trainmodel=model_type  )    
        val_loss,val_std_loss  = validate_epochs_with_dict( valid_set,model,trainmodel=model_type )
        

        if best_loss < val_loss:
            best_loss = val_loss        
            saved_dict = {'epoch': i + 1,
                         'best_acc_top1': best_loss,                         
                         'state_dict': model.state_dict(),
                         'optimizer': opt.state_dict()}
            torch.save(saved_dict,saved_modelparam_path)
            saved_epoch = i
            

        if i%10 ==0:
            print('epochs [{}/{}] | train loss {:.3f}, val loss {:.3f}, \t saved_param: {} saved at epochs {}'.format(i,nepochs,avg_loss,val_loss,saved_modelparam_path,saved_epoch ) )       
            #print('epochs [{}/{}] | current param_saved at {}'.format(saved_epoch,nepochs,saved_modelparam_path ))            
            #print(loss_list)    
        torch.cuda.empty_cache()
        
        
        
        #-------------------------------
        #wandbi log
        #-------------------------------        
        log_dict = {"tr_mnll_intrain":avg_loss,
                    "val_mnll_intrain":val_loss}        
        wandb.log(log_dict)
        
        
    #-------------------------------
    #wandbi logout for test set 
    #-------------------------------        
    #model1,_ = build_model(model_type='proposev3')
    load_dict = torch.load(saved_modelparam_path)
    model.load_state_dict(load_dict['state_dict'])           
    
    print('done')
    print('\n'*5)                                                   



