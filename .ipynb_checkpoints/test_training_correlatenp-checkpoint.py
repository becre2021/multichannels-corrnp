import wandb        
import numpy as np
from dataset_multitask_1d import motask_generator
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

    
    


def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    for ith_keys in args_dict:
        if ith_keys in ['csvpath']:
            pass
        else:
            model_info += ith_keys + str(args_dict[ith_keys]) + '_'
    return model_info


def merge_allset_1d(xc,yc,xt,yt):
    xct = torch.cat([xc,xt],dim=1)
    yct = torch.cat([yc,yt],dim=1)
    xct,s_idx =torch.sort(xct,dim=1)

    if len(xc.size()) == 3:
        yct = torch.gather(yct,1,s_idx)    
    if len(xc.size()) == 4:
        yct = torch.gather(yct,1,s_idx[:,:,0,:])
    return xct,yct
    

#from train_loss import compute_nll,compute_nll_latent    
#max_grad_norm = 1

# def scheduler_reglambda(trainprogress_ratio):
#     if trainprogress_ratio <= 0.1:
#         reglambda = 10**3    
#     if trainprogress_ratio <= 0.25:
#         reglambda = 10**2
#     elif trainprogress_ratio <= 0.5:
#         reglambda = 10**1    
#     elif trainprogress_ratio <= 0.75:
#         reglambda = 10**0    
#     else:
#         reglambda = 0    
#     return reglambda

def scheduler_reglambda(trainprogress_ratio):
    if trainprogress_ratio <= 0.25:
        reglambda = 10**4
    elif trainprogress_ratio <= 0.5:
        reglambda = 10**3    
    elif trainprogress_ratio <= 0.75:
        reglambda = 10**2    
    else:
        reglambda = 10**1    
    return reglambda

#reglamda=1.
proposed_model_list = ['gpind','gpdep']
def train_epochs_with_dict(set_dict_epoch,model,opt,lossfun,trainmodel='convcnp',trainprogress_ratio = 0.0):
    model.train()
    likelihoods = []
    
    ntask = set_dict_epoch['context_x'].size(0)
    for ith in range(ntask):
                
        context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
        target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]

        #if len(context_x.size()) == 3:
        #context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)        
        
        if model.modelname in proposed_model_list and len(context_x.size()) == 3:        
            context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)        
        
        
        # predict & train models
        #y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())    
        #obj = lossfun( y_mean,y_std, target_y.cuda())
        
        #predict & train models
        target_x,target_y = merge_allset_1d(context_x,context_y,target_x,target_y)
        y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())    
        obj = lossfun( y_mean,y_std, target_y.cuda())

        reglambda = scheduler_reglambda(trainprogress_ratio)        
        #obj += reglambda*model.compute_regloss_terms()
        #print('trainprogress_ratio,reglambda:{},{},{}'.format(trainprogress_ratio,reglambda,reglambda*model.compute_regloss_terms()))

        #print(obj,reglamda*model.compute_regloss_terms())
        
        
        obj.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        opt.step()
        opt.zero_grad()
        
        #losses.append(obj.item())
        likelihoods.append(obj.cpu().data.numpy())        
        
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       




def validate_epochs_with_dict(set_dict_epoch,model,lossfun,test_range=None,trainmodel='convcnp'):
    # large is better 
    model.eval()
    likelihoods = []
    
    ntask = set_dict_epoch['context_x'].size(0)    
    for ith in range(ntask):        
        
        context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
        target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]        
        
        if model.modelname in proposed_model_list and len(context_x.size()) == 3:                
            context_x,target_x=context_x.unsqueeze(dim=-2),target_x.unsqueeze(dim=-2)
                    
        
        
        y_mean,y_std = model(context_x.cuda(),context_y.cuda(),target_x.cuda())        
        #obj = -compute_nll( y_mean,y_std, target_y.cuda())
        obj = -lossfun( y_mean,y_std, target_y.cuda())

                    
        #if trainmodel in base_model_list:
        #    obj = -compute_nll( y_mean,y_std, target_y.cuda())
        #if trainmodel in latent_model_list:
        #    obj = -compute_nll_latent( y_mean, y_std, target_y.cuda())            
            
        likelihoods.append(obj.cpu().data.numpy())        
                
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       
    


    
    

#----------------------------------------
# argin
#----------------------------------------    
#train_range,test_range = [0,3],[3,6]   
#tasktype = 'sin3'
#tasktype = 'mosm'
#tasktype = 'lmc'

parser = argparse.ArgumentParser(description='exp1-synthetics')
parser.add_argument('--tasktype', type=str, default='sin3') # sin3,sin4,mogp,lmc,convk,
parser.add_argument('--testtype', type=str, default='extra') # inter,extra
parser.add_argument('--nepochs', type=int, default=100) #iterations
parser.add_argument('--nchannels', type=int, default=3)
parser.add_argument('--cnntype', type=str, default='shallow')
#parser.add_argument('--cnntype', type=str, default='deep')

parser.add_argument('--npostsamples', type=int, default=10)
#parser.add_argument('--ngpsamples', type=int, default=4)
#parser.add_argument('--ngpsamples', type=int, default=5)
#parser.add_argument('--ngpsamples', type=int, default=10) #mogp
parser.add_argument('--ngpsamples', type=int, default=10)

parser.add_argument('--dep', action='store_true')
parser.add_argument('--initl', type=float, default= 0.01)

#parser.add_argument('--lr', type=float, default= 0.001) #iterations
parser.add_argument('--lr', type=float, default= 0.001) #iterations
parser.add_argument('--weightdecay', type=float, default= 1e-4)


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

    
#----------------------------------------
# training 
#----------------------------------------

    
#from models.model_baseline import Convcnp_multioutput,Convcnp_multioutput2

# convcnp multiouput
# model = Convcnp_multioutput2(in_channels=3).cuda()
# opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)



num_nchannels = args.nchannels
lr = args.lr
weight_decay = args.weightdecay
init_lscale = args.initl
cnntype = args.cnntype

from test_baseline import Convcnp,compute_loss_baseline
from test_baseline_latent import  Convcnp_latent, compute_loss_baselinelatent
from test_dep_correlatenp import  DCGP_Convnp,compute_loss_gp  
from test_ind_correlatenp import  ICGP_Convnp,compute_loss_gp  



def get_model(modelname='gp'):        
    if modelname == 'base':
        num_samples = 1        
        model = Convcnp(in_dims=1,out_dims=1,num_channels=num_nchannels,init_lengthscale=init_lscale,cnntype=cnntype).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_baseline

    if modelname == 'baselatent':
        num_samples = args.npostsamples         
        model = Convcnp_latent(in_dims=1,out_dims=1,num_channels=3,num_postsamples=num_samples,init_lengthscale=init_lscale,cnntype=cnntype).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_baselinelatent
        
    if modelname == 'gpind':
        num_samples = args.ngpsamples        
        
        model = ICGP_Convnp(in_dims=1,out_dims=1,num_channels=3,num_postsamples=num_samples,init_lengthscale=init_lscale,cnntype=cnntype).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_gp
        
        
    if modelname == 'gpdep':
        num_samples = args.ngpsamples            
        model = DCGP_Convnp(in_dims=1,out_dims=1,num_channels=3,num_postsamples=num_samples,init_lengthscale=init_lscale,cnntype=cnntype).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
        lossfun = compute_loss_gp
      
    #return model,opt,lossfun    
    return model,opt,lossfun,num_samples

   
    
    
 

 
#modelnamelist = ['gp','baseline']    
#modelnamelist = ['gp','base','baselatent']    
#modelnamelist = ['gp']
#modelnamelist = ['gp_dep','gp_ind','base','base_latent']
#modelnamelist = ['gp_dep','gp_ind','base']
#modelnamelist = ['gp_ind','base','base_latent']
#modelnamelist = ['base']


#modelnamelist = ['baselatent']
#modelnamelist = ['gpdep']

#modelnamelist = ['base','baselatent']
#modelnamelist = ['gpdep','gpind']
#modelnamelist = ['gpdep']
#modelnamelist = ['gpind']
modelnamelist = ['gpdep','gpind']

torch.autograd.set_detect_anomaly(True)
for ith_model in modelnamelist:
    #model,opt,lossfun = get_model(modelname = ith_model)
    model,opt,lossfun,num_samples = get_model(modelname = ith_model)
    
        
    best_loss = -np.inf
    #saved_modelparam_path = './{}_tmp.pth'.format(model.modelname)
    #saved_modelparam_path = './param_{}/dep{}_{}_{}_initl{}.pth'.format(tasktype,dep,testtype,model.modelname,init_lscale)
    #saved_modelparam_path = './param_{}/dep{}_{}_{}_initl{}_nparam{}.pth'.format(tasktype,dep,testtype,model.modelname,init_lscale,model.num_params)
    #saved_modelparam_path = './param_{}/dep{}_{}_{}_initl{}_{}.pth'.format(tasktype,dep,testtype,model.modelname,init_lscale,cnntype)
    saved_modelparam_path = './param_{}/dep{}_{}_{}_initl{}_{}_nsamples{}.pth'.format(tasktype,dep,testtype,model.modelname,init_lscale,cnntype,num_samples)
    
    
    #-----------------------------
    #wandbi name assignment
    #-----------------------------    
    config = {'lr':lr,
              'dep':dep,
              'weight_decay':weight_decay,
              'nepochs':nepochs,
              'tasktype':tasktype,
              'testtype':testtype,
              'model':model.modelname,
              'initl':init_lscale,
              'cnntype':cnntype,
              'nsamples':num_samples}
    
        
    wandb.init( project="uai22-108",
                config = config,
                reinit= True)
    # train history
    wandb.define_metric("intestmnll_mean")    
    wandb.define_metric("intestmnll_std")    
    wandb.define_metric("outtestmnll_mean")    
    wandb.define_metric("outtestmnll_std")    
    
    #wandb.run.name = model.modelname 
    #wandb.run.name = '{}_{}_nsamples{}'.format(model.modelname,cnntype,num_samples)     
    #wandb.run.name = '{}_{}_initl{}_nsamples{}_sampler{}_v3'.format(model.modelname,cnntype,init_lscale,num_samples,model.samplertype) 
    #wandb.run.name = '{}_{}_initl{}_nsamples{}_sampler{}_v4'.format(model.modelname,cnntype,init_lscale,num_samples,model.samplertype) 
    wandb.run.name = '{}_{}_initl{}_nsamples{}_sampler{}_v5'.format(model.modelname,cnntype,init_lscale,num_samples,model.samplertype) 
    
    wandb.run.save()
    
    
    wandb.watch(model)    
    for i in range(1,nepochs + 1):   
        
        epoch_start = time.time()
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

        
        #avg_loss,std_loss = train_epochs_with_dict( train_set,model,opt,lossfun )    
        avg_loss,std_loss = train_epochs_with_dict( train_set,model,opt,lossfun, trainprogress_ratio = float(i/nepochs) )    
        
        val_loss,val_std_loss  = validate_epochs_with_dict( valid_set,model,lossfun )
        epoch_end = time.time()
        

        if best_loss < val_loss:
            best_loss = val_loss        
            saved_dict = {'epoch': i + 1,
                         'best_acc_top1': best_loss,                         
                         'state_dict': model.state_dict(),
                         'optimizer': opt.state_dict()}
            torch.save(saved_dict,saved_modelparam_path)
            saved_epoch = i


        if i%10 ==0 or i==1:
        #if i%10 ==0:            
            print('epochs [{}/{}] | train loss {:.3f}, val loss {:.3f}, \t saved_param: {} saved at epochs {} with best val loss {:.3f} \t {:.3f}(sec)'.format(i,nepochs,avg_loss,val_loss,saved_modelparam_path,saved_epoch,best_loss,epoch_end-epoch_start) )       

            #wandb.log({"tr_ll-intrain": avg_loss,
            #           "val_ll-intrain": val_loss,
            #           'current_epoch':i})
            
            if ith_model in ['gpdep','gpind']:
                wandb.log({"tr_ll-intrain": avg_loss,
                           "val_ll-intrain": val_loss,
                           'current_epoch':i,
                           'gp-regloss':model.gpsampler.regloss.cpu().data.numpy().round(2)})
                
            else:
                wandb.log({"tr_ll-intrain": avg_loss,
                           "val_ll-intrain": val_loss,
                           'current_epoch':i})

            
            
        torch.cuda.empty_cache()


        
    # eval trained params
    save_path_set = './syndata_{}/dep{}_{}_{}'.format(tasktype, dep, testtype, -1)
    print(save_path_set)
    loaded = torch.load(save_path_set + '.db')
    testset_inrange = loaded['train_set']
    testset_outrange = loaded['valid_set']
    
    load_dict = torch.load(saved_modelparam_path)
    model.load_state_dict(load_dict['state_dict'])           
    testin_loss_mean,testin_loss_std  = validate_epochs_with_dict( testset_inrange,model,lossfun )
    testout_loss_mean,testout_loss_std  = validate_epochs_with_dict( testset_outrange,model,lossfun )    
    print('modelname {}'.format(model.modelname))
    print('intest mnll: {:.2f} ({:.2f}), \t outtest mnll: {:.2f} ({:.2f})'.format(testin_loss_mean,testin_loss_std,testout_loss_mean,testout_loss_std))
    
    
    log_dict = {"intestmnll_mean":testin_loss_mean,
                "intestmnll_std":testin_loss_std,
                "outtestmnll_mean":testout_loss_mean,
                "outtestmnll_std":testout_loss_std}
                
    wandb.log(log_dict)
    
    
    
    print('\n'*5)
    
    
    
    
    
    
    
    
    
    
    
    