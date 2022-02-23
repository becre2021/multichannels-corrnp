import torch
from torch import triangular_solve,cholesky
from convcnp.utils import gaussian_logpdf
import numpy as np


def compute_multioutput_K(k_sub,x1,x2=None,eps=5e-4):
    if x2 is None:
        x2 = x1
        x2_none = True
    else:
        x2_none = False
    output_dims = x1.size(1)

    K = []
    k_dict = {}    
    if x2_none:
        for i in range(output_dims):
            k_i = []
            for j in range(output_dims):
                if i==j:
                    ksub_ij = k_sub(i,i,x1[:,i],x2[:,i])          
                    if ksub_ij.is_cuda and x2_none:
                        ksub_ij += eps*torch.eye(x1[:,i].size(0)).cuda()
                    else:
                        ksub_ij += eps*torch.eye(x1[:,i].size(0))
                elif i<j:
                    ksub_ij = k_sub(i,j,x1[:,i],x2[:,j])                
                    k_dict[(i,j)] = ksub_ij
                else:
                    ksub_ij = k_dict[(j,i)].T
                k_i.append(ksub_ij)
            K.append(torch.cat(k_i,dim=-1))
    else:
        for i in range(output_dims):
            k_i = []
            for j in range(output_dims):
                if i==j:
                    ksub_ij = k_sub(i,i,x1[:,i],x2[:,i])          
                else:
                    ksub_ij = k_sub(i,j,x1[:,i],x2[:,j])                
                    k_dict[(i,j)] = ksub_ij
                k_i.append(ksub_ij)
            K.append(torch.cat(k_i,dim=-1))
        
    del k_dict
    K = torch.cat(K,dim=0)
    return K


def gp_predict_batch(context_x,context_y,target_x, Ksub, diag = True):

    nb,ntarget,nchannel = target_x.size()
    
    b_mean,b_var = [],[]
    for i_context_x,i_context_y,i_target_x in zip(context_x,context_y,target_x):
    
        #K_cc = compute_multioutput_K(gen_cls.f.Ksub,i_context_x)
        #K_ct = compute_multioutput_K(gen_cls.f.Ksub,i_context_x,i_target_x)
        #K_cc = compute_multioutput_K(gen_cls.f.Ksub,i_context_x)
        #K_ct = compute_multioutput_K(gen_cls.f.Ksub,i_context_x,i_target_x)
        K_cc = compute_multioutput_K(Ksub,i_context_x)
        K_ct = compute_multioutput_K(Ksub,i_context_x,i_target_x)
        
        i_c_y = i_context_y.transpose(0,1).reshape(-1,1)
        #i_t_y = i_target_y.transpose(0,1).reshape(-1,1)
        
        L = cholesky(K_cc)
        A = triangular_solve(K_ct,L,upper=False)[0]   #A = trtrs(L, k_xs)
        V = triangular_solve(i_c_y,L,upper=False)[0]

        mean_f = torch.mm(torch.transpose(A, 0, 1), V)        
        if diag:
            #var_f1 = compute_multioutput_K(gen_cls.f.Ksub,i_target_x).diag()         
            var_f1 = compute_multioutput_K(Ksub,i_target_x).diag()         

            var_f2 = torch.sum(A * A, 0)

            
            #var_f = (var_f1 - var_f2).reshape(-1,1) +  self.likelihood.variance.transform()**2
            var_f = (var_f1 - var_f2).reshape(-1,1) 
            #return mean_f, (var_f1 - var_f2).reshape(-1,1) +  self.likelihood.variance.transform()**2
            
        else:
            #var_f1 = compute_multioutput_K(gen_cls.f.Ksub,i_target_x)            
            var_f1 = compute_multioutput_K(Ksub,i_target_x)            
            
            var_f2 = torch.mm(A.t(), A)
            #var_f = (var_f1 - var_f2) +  (self.likelihood.variance.transform()**2).diag().diag()
            var_f = (var_f1 - var_f2) 
            #return mean_f, (var_f1 - var_f2) +  (self.likelihood.variance.transform()**2).diag().diag()

        b_mean.append(mean_f.view(nchannel,-1).T[None,:,:])
        b_var.append(var_f.view(nchannel,-1).T[None,:,:])
        
#     b_mean = torch.cat(b_mean,dim=1).T.reshape(nb,-1,nchannel)        
#     b_std = torch.cat(b_var,dim=1).T.reshape(nb,-1,nchannel).sqrt()        
    b_mean = torch.cat(b_mean,dim=0)    
    if diag:
        b_std = torch.cat(b_var,dim=0).sqrt()
    else:
        raise NotImplementedError
        
    return b_mean,b_std
        
    
    
    
    
def validate_oracle_epochs_with_dict(set_dict_epoch,Ksub=None,test_range=None):
    # large is better 
    #model.eval()
    likelihoods = []
    
    ntask = set_dict_epoch['context_x'].size(0)    
    for ith in range(ntask):        
        
        context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
        target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]
        #y_mean,y_std = gp_predict_batch(context_x.cuda(),context_y.cuda(),target_x.cuda(),Ksub=Ksub)
        y_mean,y_std = gp_predict_batch(context_x,context_y,target_x,Ksub=Ksub)
        
        obj = gaussian_logpdf(target_y, y_mean, y_std, 'batched_mean')        
        #obj = gaussian_logpdf(target_y.cuda(), y_mean, y_std, 'batched_mean')        
        likelihoods.append(obj.cpu().data.numpy())        
                
    avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
    return avg_ll,std_ll       



