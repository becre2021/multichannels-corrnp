# see example for wandb : https://github.com/wandb/examples
import wandb        
import os
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import argparse
import time 
import random
random_seed = 1111
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
    


#----------------------------------------
# argin
#----------------------------------------    
#train_range,test_range = [0,3],[3,6]   
#tasktype = 'sin3'
#tasktype = 'mosm'
#tasktype = 'lmc'

parser = argparse.ArgumentParser(description='exp1-synthetics-singletask')
parser.add_argument('--modelname',type=str, default='base') #--modelname gpdep2
parser.add_argument('--tasktype', type=str, default='singletask') # sin3,sin4,mogp,lmc,convk,
parser.add_argument('--testtype', type=str, default='extra') # inter,extra
parser.add_argument('--dep', action='store_true')
parser.add_argument('--cnntype', type=str, default='shallow')

parser.add_argument('--nepochs', type=int, default=100) #iterations
parser.add_argument('--nchannels', type=int, default=1)
#parser.add_argument('--cnntype', type=str, default='deep')

#parser.add_argument('--npostsamples', type=int, default=10)
#parser.add_argument('--npostsamples', type=int, default=3)
parser.add_argument('--npostsamples', type=int, default=10)
parser.add_argument('--ngpsamples', type=int, default=10)

parser.add_argument('--initl', type=float, default= 0.1)
parser.add_argument('--lr', type=float, default= 1e-3) #iterations

parser.add_argument('--weightdecay', type=float, default= 1e-4)

parser.add_argument('--datav',type=int, default=3) #--modelname gpdep2
parser.add_argument('--printfreq',type=int, default=100) #--modelname gpdep2
parser.add_argument('--runv',type=int, default=15) #--modelname gpdep2
parser.add_argument('--reglambda',type=float, default=1.) #--modelname gpdep2
parser.add_argument('--msg',type=str,default='none')







def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    for ith_keys in args_dict:
        if ith_keys in ['csvpath']:
            pass
        else:
            model_info += ith_keys + str(args_dict[ith_keys]) + '_'
    return model_info


args = parser.parse_args()   
expinfo = make_expinfo(args)
print('#'+'-'*100)
print(expinfo[:-1])
print('#'+'-'*100)
print('\n')

tasktype = args.tasktype
testtype = args.testtype
nepochs = args.nepochs
dep = args.dep



    
    




import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence as kldiv
from torch.distributions.normal import Normal


def compute_regloss(yt,outs,nllloss,eps=1e-6,tempering0=1e0):
    
    #------------------------------
    #kl div
    #------------------------------    
    posterior_target=outs.gpouts.posterior_target  
    #tempering0 = 
    nb,_,ntarget,nmixtures,nchannels = posterior_target.shape
    
    pmu = posterior_target.mean(dim=1)
    pstd = posterior_target.std(dim=1)
    normal_dist = Normal(loc=pmu, scale=pstd)                         
    #logprob = normal_dist.log_prob(yt[:,:,None,:]).mean(dim=1)
    emp_logprob = normal_dist.log_prob(yt[:,:,None,:]).mean(dim=1) #(nb,nmixtures,nchannel)
    emp_logprob = emp_logprob - emp_logprob.max(dim=1,keepdim=True)[0]
    appr_logits = F.softmax(emp_logprob/tempering0,dim=1).detach().clone()    
    appr_logits = appr_logits.permute(0,2,1)
    outs.labels = appr_logits    

    learnable_logits = outs.gpouts.neural_logits

    
    
    #eps=1e-6
    q_cat = Categorical(learnable_logits+eps)
    p_cat = Categorical(appr_logits+eps)
    wregloss = kldiv(q_cat,p_cat).sum(dim=-1).mean()  #(mb,nchannel,nmixture) -> (1)  
    
    #scale = 0.5*np.abs(nllloss.item()/(wregloss.item() + eps))
    #scale = 1.*np.abs(nllloss.item()/(wregloss.item() + eps))
    scale = 0.5*np.abs(nllloss.item()/(wregloss.item() + eps))
    
    
    outs.regloss = scale*wregloss 
    return outs
    


    
def merge_allset_1d(xc,yc,xt,yt):
    xct = torch.cat([xc,xt],dim=1)
    yct = torch.cat([yc,yt],dim=1)
    xct,s_idx =torch.sort(xct,dim=1)

    if len(xc.size()) == 3:
        yct = torch.gather(yct,1,s_idx)    
    if len(xc.size()) == 4:
        yct = torch.gather(yct,1,s_idx[:,:,0,:])
    return xct,yct






proposed_model_list = ['gpind','gpdep']
def train_epochs_pair(batch_dataset_pair,model,opt,lossfun,current_iter=1,total_iter = 500):    
    
    model.train()
    opt.zero_grad()
    
    
    likelihoods,regloss = [],[]
    for dataset_pair in batch_dataset_pair:
        #context_x,context_y,target_x,target_y = dataset_pair        
        context_x,context_y,target_x,target_y = dataset_pair[:4]        
        
        if model.modelname in proposed_model_list and len(context_x.size()) == 3:        
            context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)        
            
            
        target_x,target_y = merge_allset_1d(context_x,context_y,target_x,target_y)

        
        #predict & train models
        outs = model(context_x.cuda(),context_y.cuda(),target_x.cuda())    
        obj = lossfun( outs.pymu, outs.pystd, target_y.cuda(), intrain=True) 
        
    
        if model.modelname in ['gpdep']:
            outs = compute_regloss(target_y.cuda(),outs,obj)
            lambda_regloss = outs.regloss             
            obj = obj + lambda_regloss
            regloss.append(lambda_regloss.cpu().data.numpy())       
                
        likelihoods.append(obj.cpu().data.numpy())                                       
        obj.backward()
        opt.step()
        opt.zero_grad()
        
        
        if model.modelname in ['gpdep']:
            model.gpsampler.bound_hypparams()
            
        
    ntask = len(batch_dataset_pair)
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    avg_reg,std_reg = np.array(regloss).mean().round(2),(np.array(regloss).std()/np.sqrt(ntask)).round(2)
    return avg_ll,avg_reg




        



def validate_epochs_pair(batch_dataset_pair,model,lossfun):
    # large is better 
    model.eval()
    likelihoods = []
    
    #ntask = set_dict_epoch['context_x'].size(0)    
    #for _ in range(ntask):        
    ntask = len(batch_dataset_pair)
    for dataset_pair in batch_dataset_pair:
        context_x,context_y,target_x,target_y,full_x,full_y = dataset_pair        
        if model.modelname in proposed_model_list and len(context_x.size()) == 3:                
            context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)


        #y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())
        outs  = model(context_x.cuda(),context_y.cuda(),target_x.cuda())
        #obj = -lossfun( y_mean,y_std, target_y.cuda(), intrain=False)        
        obj = -lossfun( outs.pymu, outs.pystd, target_y.cuda(), intrain=True)        
        
        likelihoods.append(obj.cpu().data.numpy())        

    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll     
    


    


    
    
    
#----------------------------------------
# training 
#----------------------------------------
from models.test_baseline import Convcnp,compute_loss_baseline
from models.test_baseline_latent import  Convcnp_latent, compute_loss_baselinelatent
from models.test_dep_correlatenp import  DCGP_Convnp,compute_loss_gp  
from models.test_ind_correlatenp import  ICGP_Convnp,compute_loss_gp  
from models.test_cnp import RegressionANP, RegressionCNP




# lr = args.lr
# weight_decay = args.weightdecay
init_lscale = args.initl
def get_model(modelname='gp',cnntype = 'shallow'):        
    num_nchannels = args.nchannels

    
    if modelname == 'base':
        num_samples = 1        
        model = Convcnp(in_dims=1,out_dims=1,num_channels=num_nchannels,init_lengthscale=init_lscale,cnntype=cnntype).cuda()        
        lossfun = compute_loss_baseline
        
    if modelname == 'baselatent':
        num_samples = args.npostsamples         
        model = Convcnp_latent(in_dims=1,out_dims=1,num_channels=num_nchannels,num_postsamples=num_samples,init_lengthscale=init_lscale,cnntype=cnntype).cuda()
        lossfun = compute_loss_baselinelatent
        
    if modelname == 'gpind':
        num_samples = args.ngpsamples                
        model = ICGP_Convnp(in_dims=1,
                            out_dims=1,
                            num_channels=num_nchannels,
                            num_postsamples=num_samples,
                            init_lengthscale=init_lscale,
                            cnntype=cnntype).cuda()       
        lossfun = compute_loss_gp
        
        
    if modelname == 'gpdep':
        num_samples = args.ngpsamples            
        model = DCGP_Convnp(in_dims=1,
                            out_dims=1,
                            num_channels=num_nchannels,
                            num_postsamples=num_samples,
                            init_lengthscale=init_lscale,
                            cnntype=cnntype).cuda()
        lossfun = compute_loss_gp
        
        
 
    if modelname == 'anp':
        num_samples = 1        
        model = RegressionANP(input_dim=num_nchannels,
                              latent_dim=128,
                              num_channels=num_nchannels).cuda()   
        
        lossfun = compute_loss_baseline
        
        
    if modelname == 'cnp':
        num_samples = 1                
        model = RegressionCNP(input_dim=num_nchannels,
                              latent_dim=128,
                              num_channels=num_nchannels).cuda()       
        lossfun = compute_loss_baseline

      
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = args.weightdecay)                
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max= args.nepochs)    
    return model,opt,lossfun,num_samples,scheduler

   
    
    
 



#----------------------------------------
# model instance 
#----------------------------------------
model,opt,lossfun,num_samples,scheduler = get_model(modelname = args.modelname , cnntype = args.cnntype )




save_dir = './regression_task_single/param_{}/'.format(tasktype)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_filename = 'dep{}_{}_{}_initl{}_{}_nsamples{}_nepochs{}_datav{}_gppriorscale{}_runv{}_reglam{}.pth'.format(dep,
                                                                                                       testtype,
                                                                                                       args.modelname,
                                                                                                       init_lscale,
                                                                                                       args.cnntype,
                                                                                                       num_samples,
                                                                                                       args.nepochs,
                                                                                                       args.datav,
                                                                                                       model.gppriorscale,
                                                                                                       args.runv,
                                                                                                       args.reglambda)
saved_modelparam_path = save_dir + save_filename


#-----------------------------
#wandbi name assignment
#-----------------------------    
config = {'lr':args.lr,
          'dep':args.dep,
          'weight_decay':args.weightdecay,
          'nepochs':nepochs,
          'tasktype':tasktype,
          'testtype':testtype,
          'model':model.modelname,
          'initl':init_lscale,
          'cnntype':args.cnntype,
          'nsamples':num_samples,
          'datav':args.datav,
          'runv':args.runv,
          'reglam':args.reglambda}

wandb.init( project="uai22-9234-synthetic-singlemixedtask",
            notes="datav{}, msg:{}".format(args.datav,args.msg),            
            config = config,           
            reinit= True)



wandb.run.name = '{}_{}_initl{}_nsamples{}_gpprior{}_datav{}_runv{}_reglam{}'.format(args.modelname,args.cnntype,init_lscale,num_samples,model.gppriorscale,args.datav,args.runv,args.reglambda)    #--deep back and use actiavtion


best_loss = -np.inf
wandb.run.save()
wandb.watch(model)


#with torch.autograd.detect_anomaly():
torch.autograd.set_detect_anomaly(True)
for i in range(1,args.nepochs + 1):   

    epoch_start = time.time()
    try:
        save_path_set = './regression_task_single/syndata_{}_v{}/dep{}_{}_{}'.format(tasktype,args.datav, dep, testtype, i)

        loaded = torch.load(save_path_set + '.db')
        train_set = loaded['train_set']
    except:
        print('failed load at {}'.format(save_path_set ))            
        pass

    avg_loss,avg_reg = train_epochs_pair( train_set,model,opt,lossfun, current_iter = i ,total_iter = args.nepochs )        
    epoch_end = time.time()


    #if i%1 ==0:
    if i%args.printfreq ==0 or i== 1:                
        save_path_set = './regression_task_single/syndata_{}_v{}/dep{}_{}_{}'.format(tasktype,args.datav, dep, testtype, -128)
        loaded = torch.load(save_path_set + '.db')
        batch_intrain_pair = loaded['train_set']
        val_loss,_ = validate_epochs_pair( batch_intrain_pair,model,lossfun )
        if best_loss < val_loss:
            best_loss = val_loss        
            saved_dict = {'epoch': i + 1,
                         'best_acc_top1': best_loss,                         
                         'state_dict': model.state_dict(),
                         'optimizer': opt.state_dict(),
                         'scheduler':scheduler.state_dict(),
                         'start_epochs':args.nepochs }
            torch.save(saved_dict,saved_modelparam_path)
            saved_epoch = i




        print('epochs [{}/{}] | train loss {:.3f}, val loss {:.3f}, \t saved_param: {} saved at epochs {} with best val loss {:.3f} \t {:.3f}(sec)'.format(i,nepochs,avg_loss,val_loss,saved_modelparam_path,saved_epoch,best_loss,epoch_end-epoch_start) )                       
        #wandbi tarinining check
        wandb.log({"tr_ll-intrain": avg_loss,"val_ll-intrain": val_loss,'current_epoch':i,'tr_reg-intrain':avg_reg})


    torch.cuda.empty_cache()



print('\n'*5)



