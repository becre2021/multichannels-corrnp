from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import itertools as it


from convcnp.utils import to_multiple
from test_kernels import pi2repuse


from attrdict import AttrDict
from cholesky_save import psd_safe_cholesky


#__all__= ['Spikeslab_GPsampler']
__all__= ['Independent_GPsampler']

    
    
pi2 = 2*math.pi
eps=1e-6
pi2repuse=True
def eval_smkernel_batch(param_list,xc,xt=None, likerr_bound=[.1,1.0], zitter=1e-4):
#def eval_smkernel(param_list,xc,xt=None, likerr_bound=[.1,1.0], zitter=1e-3,zitter_flag = False):

    """
    inputs:
        #xc : (nb,ncontext,ndim,nchannel)
        xc : (nb,ncontext,ndim)
        
    outputs:
        Kxz : (nb,ncontext,ncontext,nchannel)      # assume y-value is 1-d      
    """
    
    assert len(xc.shape) == 4                            
    if xt is None:
        xt = xc
        nb,ndata,ndim,nchannel = xc.size()         
        ndata2=ndata
    else:
        nb,ndata,ndim,nchannel = xc.size()                            
        _,ndata2,_,_ = xt.size()                            

    xc_ = xc.unsqueeze(dim=1)
    xt_ = xt.unsqueeze(dim=1)
              
    assert len(param_list) == 4    
    
#     #(nmixutre,ndim),(nmixutre,ndim),(nmixutre),(1)
#     mu,inv_std,logits,likerr = param_list         
#     mu_=mu[None,:,None,:,None]
#     inv_std_=inv_std[None,:,None,:,None] 

    #(nchannel,ndim),(nchannel,ndim),(nchannel),(1)
    mu,inv_std,logits,likerr = param_list         
    #mu_=mu[None,:,None,:,None]
    #inv_std_=inv_std[None,:,None,:,None] 
    mu_=mu.permute(1,0)[None,None,:,:]
    inv_std_=inv_std.permute(1,0)[None,None,:,:]
    
    #(nb,ndata,ndim,nchannel)
    xc_ = xc
    xt_ = xt
    #xc_.shape,xt_.shape

    exp_xc_ = xc_*inv_std_
    exp_xt_ = xt_*inv_std_
    cos_xc_ = xc_*mu_
    cos_xt_ = xt_*mu_

    
    #(nb,ndata,nchannel)
    exp_term_xc2_ = torch.pow(exp_xc_,2).sum(dim=-2)[:,:,None,:] 
    exp_term_xt2_ = torch.pow(exp_xt_,2).sum(dim=-2)[:,None,:,:]
    cross_term_ = torch.einsum('badk,bcdk->back',exp_xc_,exp_xt_)    
    exp_term = exp_term_xc2_ + exp_term_xt2_ -2*cross_term_    
    cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:]
    
    if pi2repuse:    
        weighted_outs = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
    else:
        weighted_outs = torch.exp(-0.5*exp_term )*torch.cos(cos_term)        
        
        
    
    if ndata == ndata2: 
        likerr_ = likerr[None,None,None,:]
        likerr_ = torch.clamp(likerr_,min=likerr_bound[0],max=likerr_bound[1])                
        #print(likerr_.shape)
        noise_eye = (zitter+likerr_**2)*(torch.eye(ndata)[None,:,:,None]).to(xc.device)
        weighted_outs = weighted_outs + noise_eye     
        return weighted_outs
    
    else:
        return weighted_outs
    
    


#num_mixtures = 3
#num_mixtures = 5
#num_mixtures = 10

#class NeuralSpikeslab_GPsampler(nn.Module):        
#class Spikeslab_GPsampler(nn.Module):     



class Independent_GPsampler(nn.Module):           
    def __init__(self,in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 10,num_sampleposterior=10 ,
                      scales=.5, loglik_err=1e-2, eps=1e-6,points_per_unit=64,multiplier=2**3 ):
        
        super(Independent_GPsampler, self).__init__()        
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_channels = num_channels
        self.num_fourierbasis = num_fourierbasis
        #self.kernel = kernel
        #self.target_idx = self.kernel.target_idx                
        #self.cross_idx = self.kernel.cross_idx                
        
        self.normal0 = Normal(loc=0.0,scale=1.0)
        self.uniform0 = Uniform(0,1)
            
        self.points_per_unit = points_per_unit  
        self.multiplier = multiplier        
        #self.inverseuse = inverseuse
        self.regloss = 0.0
        self.num_mixtures = 1
        assert self.num_mixtures == 1
        self.set_initparams(scales=scales,loglik_err = loglik_err,eps =eps)
        
        
                    
        self.w=None
        self.b=None        
        self.normalizer=None
        self.random_w=None        
        #self.prior_scale = 0.5        
        self.prior_scale = 1.0              
        self.temperature= 1e-3        
        
        self.use_constrainedprior = False #True means density aware prior 
        #self.use_constrainedprior = True #True means density aware prior 
        
        return 

    
    
    def set_initparams(self,scales=1.0,loglik_err=1e-2,eps=1e-6):
        # learnable parameters
        #loglogits = eps  +  1.*torch.rand(self.num_channels,self.num_mixtures)
        #loglogits = eps  +  1.*torch.ones(self.num_channels,self.num_mixtures)
        #loglogits +=  0.1*torch.rand(self.num_channels,self.num_mixtures)
        
#         #logmu = eps  +  1.*torch.rand(self.num_mixtures,self.in_dims)                
#         logmu = eps  +  eps*torch.rand(self.num_mixtures,self.in_dims)   #rbf kerenl             
#         logmu[0] = eps*torch.ones(self.in_dims) 
#         logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
#         loglik = eps + loglik_err*torch.ones(self.num_channels)



        #---------------------------------
        # single-channl tasks in section 5-1    
        #---------------------------------
        
        if self.num_channels == 1:          
#             # -------------------------
#             # runv 1 - regraded as sm prior
#             # -------------------------            
#             logmu = eps  +  1*torch.rand(self.num_channels,self.in_dims)               
#             logmu[0] = eps*torch.ones(self.in_dims) 
#             logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
#             loglik = eps + loglik_err*torch.ones(self.num_channels)

#             # -------------------------
#             # runv 2 - regraded as rbf prior
#             # -------------------------            
#             logmu = eps  +  eps*torch.rand(self.num_channels,self.in_dims)               
#             logmu[0] = eps*torch.ones(self.in_dims) 
#             logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
#             loglik = eps + loglik_err*torch.ones(self.num_channels)

            # -------------------------
            # runv 3 - regraded as sm prior - re
            # -------------------------                        
            maxfreq = 5.
            logmu = eps  +  maxfreq*torch.rand(self.num_channels,self.in_dims)               
            #logmu[0] = eps*torch.ones(self.in_dims) 
            logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
            loglik = eps + loglik_err*torch.ones(self.num_channels)
            
            #print('logmu')
            #print(logmu)
            #init as expected
        
        #---------------------------------
        # multi-channl tasks in section 5-2
        #---------------------------------        
        if self.num_channels == 3:        
            
            #----------------------------
            # sin3 task 
            #----------------------------    
#             # regraded as sm prior           
#             logmu = eps  +  1*torch.rand(self.num_channels,self.in_dims)               
#             logmu[0] = eps*torch.ones(self.in_dims) 
#             logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
#             loglik = eps + loglik_err*torch.ones(self.num_channels)

            # regraded as rbf prior            
            logmu = eps  +  eps*torch.rand(self.num_channels,self.in_dims)               
            logmu[0] = eps*torch.ones(self.in_dims) 
            logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
            loglik = eps + loglik_err*torch.ones(self.num_channels)

 

#             #----------------------------
#             # mosm task 
#             #----------------------------    
#             # regraded as sm kernel prior           
#             maxfreq = 5.
#             logmu = eps  +  maxfreq*torch.rand(self.num_channels,self.in_dims)               
#             logmu[0] = eps*torch.ones(self.in_dims) 
#             logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
#             loglik = eps + loglik_err*torch.ones(self.num_channels)

#             # regraded as rbf kernel prior
#             logmu = eps  +  1*torch.rand(self.num_channels,self.in_dims)               
#             logmu[0] = eps*torch.ones(self.in_dims) 
#             logstd = eps + scales*torch.ones(self.num_channels,self.in_dims)        
#             loglik = eps + loglik_err*torch.ones(self.num_channels)


        
        self.logmu =    nn.Parameter( torch.log( logmu )) 
        self.logstd =   nn.Parameter( torch.log( logstd ))                  
        self.loglik =   nn.Parameter( torch.log( loglik ))
        self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
        return 
    
    
    
    def bound_hypparams(self,bound_std = [1.,2.]):
        pass
        #bound_logstd = np.log(bound_std)
        #with torch.no_grad():
        #    self.logstd.data.clip_(bound_logstd[0],bound_logstd[1])
        #    
        #print('param_list')
        #print(self.param_list)
        return
    
    
    
    def build_xgrid(self,xc,xt,x_thres=1.0):
        nb,_,ndim,nchannel=xc.size()         
        x_min = min(torch.min(xc).cpu().numpy(),torch.min(xt).cpu().numpy()) - x_thres
        x_max = max(torch.max(xc).cpu().numpy(),torch.max(xt).cpu().numpy()) + x_thres
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))                     
        xgrid = torch.linspace(x_min,x_max, num_points)
        xgrid = xgrid.reshape(1,num_points,1).repeat(nb,1,ndim).to(xc.device)
        return xgrid
        
    

  
    #def sample_w_b(self,nb,eps=1e-6):    
    #def sample_w_b(self,nb,nsamples,eps=1e-6):    
    def sample_w_b(self,xc,yc,nsamples,eps=1e-6,tempering=10.):    
        
        """
        self.w_mu : nparams
        sample_w  : (nb,nfourifbasis,indim,nchannels)
        sample_b  : (nb,nfouribasis,indim,nchannels)        
        """        
        nb = xc.size(0)
        
        # (num_mixtures,num_dim)
        mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        
        
        assert self.num_mixtures == 1        
        eps1 = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_channels,self.in_dims)).to(mu.device)
        eps2 = self.uniform0.sample((nb,nsamples*self.num_fourierbasis,self.num_channels,1)).to(mu.device)           
        
        #print('allow channel dependency')
        #random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,1)).to(mu.device) #impose depedency over channels
        #random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,self.num_channels)).to(mu.device) #impose depedency over channels
        random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_channels)).to(mu.device) #impose depedency over channels        
        sample_w = mu[None,None,:,:] + inv_std[None,None,:,:]*eps1    #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)                
        sample_b = eps2                                                #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)

        
        return sample_w,sample_b,random_w
        #return sample_w,sample_b,random_w,logits_samples
    
    
    
    
    #def sample_prior_shared(self,xc,xt,numsamples=10,reorder=False, temperature= 1e-3):        
    #def sample_prior_shared(self,xc,xt,numsamples=10,reorder=False):        
    def sample_prior_shared(self,xc,yc,xt,numsamples=10,reorder=False):        
        
        """
        inputs:
            xc : (nb,ncontext,ndim,nchannel)
            xt : (nb,ntarget,ndim,nchannel)        
        outputs:
            xa_samples : (nb,nchannel*(ncontext+ntarget),ndim)
            Psi_pred    : (nb,nchannel*(ncontext+ntarget),nchannel)      # assume y-value is 1-d      
        """
        nb = xc.size(0)
        #if xt in None:
        #xa_samples = self.samples_xa(xc,xt)                                                 #(nb,nchannel*(ncontext+ntarget),ndim)  
        xa_samples =self.build_xgrid(xc,xt)
        w,b,random_w = self.sample_w_b(xc,yc,numsamples)    
        
        #print('logits_samples.shape')
        #print(logits_samples.shape)
        
        self.w = w
        self.b = b
        self.random_w = random_w   
        
        
        
        #(nb,nsamples*nfourier,in_dim,num_channels)
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)
        
        #(nb,ndata2,1)--> (nb,nsamples*nfourier,ndata2,1)        
        xa_samples_ = xa_samples[:,None,:].repeat(1,w.size(1),1,1)
        #(nb,nsamples*nfourier,ndata2,num_channels)                
        xadotw = torch.einsum('bjxy,bjyz->bjxz',xa_samples_,w)
        
        if pi2repuse:            
            cos_interms = pi2*xadotw + pi2*b                   
        else:
            cos_interms = xadotw + pi2*b                 

        Psi = torch.cos(cos_interms)            
        nb,_,ndata2,_ = Psi.shape

        #(nb,nsamples*self.num_fourierbasis,self.num_channels)
        random_w = random_w[:,:,None,:]
        sum_costerm =  (Psi*random_w).reshape(nb,numsamples,self.num_fourierbasis,ndata2,self.num_channels)   #self.num_mixtures=1     
        sum_costerm = sum_costerm.sum(dim=2)

        #(nb,nsamples,ndata2,self.num_channels)        
        normalizer = np.sqrt(2/self.num_fourierbasis)        
        prior_samples = normalizer*sum_costerm        
        return prior_samples,xa_samples                         
        
        #logits_samples_ = logits_samples.permute(0,1,3,2)[:,:,None,:,:]
        #w_prior_samples = (prior_samples*logits_samples_).sum(dim=-2)                
        #return w_prior_samples,xa_samples                 


 

    #def sample_prior_independent(self,xc,yc,numsamples=10,newsample=False):            
    def sample_prior_independent(self,xc,numsamples=10,newsample=False):            
        
        nb,ndata,ndim,nchannel = xc.shape
        w,b,random_w = self.w, self.b,self.random_w
            
        #(nb,nsamples*nfourier,in_dim,num_channels)            
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)

        #(nb,nsamples*nfourier,ndata,ndim,num_channel)        
        xc_ = xc[:,None,:,:,:].repeat(1,w.size(1),1,1,1)
        w_ = w[:,:,:,None,:]  #(nb,nsamples*nfourier,in_dim,1,num_channels)     
        b_ = b[:,:,:,None,:]

        if pi2repuse:            
            xcdotw_b = pi2*(torch.einsum('bsndc,bsdmc->bsnmc',xc_,w_) + b_)
        else:
            xcdotw_b = torch.einsum('bsndc,bsdmc->bsnmc',xc_,w_) + pi2*b_
        if xcdotw_b.size(-2) == 1:
            xcdotw_b = xcdotw_b.sum(dim=-2) #(nb,nsamples*nfourier,ndata,num_channels) 
        
        
        Psi = torch.cos(xcdotw_b)        
        sum_costerm = Psi*random_w[:,:,None,:]
        #sum_costerm_ = sum_costerm.reshape(nb,numsamples,-1,ndata,self.num_mixtures,nchannel)
        sum_costerm_ = sum_costerm.reshape(nb,numsamples,-1,ndata,self.num_channels)
        
        normalizer = np.sqrt(2/self.num_fourierbasis)
        prior_samples = normalizer*(sum_costerm_.sum(dim=2))
        return prior_samples
    


    def prepare_updateterms(self,xc,yc,xa_shared=None,xt=None,numsamples=1):
        nb,ndata,ndim,nchannel = xc.shape

        likerr = self.loglik.exp()
        likerr_bound = self.loglik_bound

        mu = self.logmu.exp()
        inv_std  = 1/(self.logstd.exp()+eps)
        #logits = F.softmax(self.loglogits,dim=-1)        
        #logits = F.softmax(self.neural_loglogits,dim=-1)                
        logits = None
        param_list = (mu,inv_std,logits,likerr)
        
        
        
        xa_shared_ = xa_shared[...,None].repeat(1,1,1,nchannel)
        K_cc = eval_smkernel_batch(param_list,xc,  likerr_bound=likerr_bound)                
        K_ac = eval_smkernel_batch(param_list,xa_shared_, xc,  likerr_bound=likerr_bound)
        #print('K_cc.shape,K_ac.shape')
        #print(K_cc.shape,K_ac.shape)
        
        K_cc_ = K_cc.permute(0,3,1,2) 
        K_ac_ = K_ac.permute(0,3,1,2) #(nb,nchannel,ndata2,ndata)
        #K_cc_.shape,K_ac_.shape
        L = torch.linalg.cholesky(K_cc_ )          
        #w_prior_ind = self.sample_prior_independent(xc,xt,numsamples=numsamples)        
        w_prior_ind = self.sample_prior_independent(xc,numsamples=numsamples)                
        w_prior_ind =  w_prior_ind +  likerr[None,None,None,:]*torch.randn_like(w_prior_ind).to(xc.device)
        density_term = K_ac_.sum(dim=-1).permute(0,2,1)    
        
        if self.use_constrainedprior:            
            density_term2 = K_ac_.sum(dim=-2).permute(0,2,1)   #(nb,ndata,nhcannel)            
            allow_prior = 1. - 2*(torch.sigmoid(density_term2/0.1)-.5) + eps #(nb,ndata2,nchannels)
            allow_prior = allow_prior[:,None,:,:].detach().clone()
            w_prior_ind = allow_prior*w_prior_ind 
            
        
        delta_yc = yc[:,None,:,:]  - w_prior_ind
        delta_yc = delta_yc.permute(0,3,2,1)
        Kinvyc = torch.cholesky_solve(delta_yc,L,upper=False)        
        update_term_shared = torch.einsum('bnac,bncs->bnas',K_ac_,Kinvyc).permute(0,3,2,1)        
        
        
        K_tc = eval_smkernel_batch(param_list, xt.clone(), xc.clone(),  likerr_bound=likerr_bound) 
        #K_tc = eval_smkernel_batch(param_list, xt.clone(), xc.clone(),  likerr_bound=likerr_bound)
        K_tc_ = K_tc.permute(0,3,1,2)        
        update_term_target = torch.einsum('bnac,bncs->bnas',K_tc_,Kinvyc).permute(0,3,2,1)        
        
        
        #return update_term_shared,density_term,update_term_target 
        return update_term_shared,density_term,update_term_target 
    
    
    

    #def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None,use_constrainedprior=False):
    def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None,use_constrainedprior=True):
        
        #prior_shared, xa_shared = self.sample_prior_shared(xc,xt,numsamples=numsamples)
        prior_shared, xa_shared = self.sample_prior_shared(xc,yc,xt,numsamples=numsamples)
                
        #update_term,density_term = self.prepare_updateterms(xc,yc,xa_shared=xa_shared,numsamples=numsamples)
        update_term_shared, density_term, update_term_target = self.prepare_updateterms(xc,yc,xa_shared=xa_shared,xt=xt,numsamples=numsamples)               
                
        if self.use_constrainedprior:        
            allow_prior = 1. - 2*(torch.sigmoid(density_term/0.1)-.5) + eps #(nb,ndata2,nchannels)
            allow_prior = allow_prior[:,None,:,:].detach().clone()
            prior_shared  = prior_shared*allow_prior
        
        posterior_shared =  prior_shared + update_term_shared
                
        prior_target = self.sample_prior_independent(xt,numsamples=numsamples)                
        posterior_target = prior_target + update_term_target
        
        outs = AttrDict()
        outs.xa_samples = xa_shared    #(nb,ndata2,ndim)                  
        outs.prior_samples =  prior_shared      
        outs.posterior_samples= posterior_shared         
        #outs.posterior_samples= posterior_shared.detach().clone()  #it is not good (validated)        
        outs.posterior_target= posterior_target 
        
        outs.density = density_term 
        outs.regloss = 0.0        
        return outs    

    
    
    

    
#     #def sample_prior_independent(self,xc,numsamples=10):    
#     def sample_prior_independent_loop(self,xc,yc,numsamples=10):    
        
#         """
#         inputs:
#             xc : (nb,ncontext,ndim,nchannel)
#         outputs:
#             Psi_pred : (nb,ncontext,nchannel)      # assume y-value is 1-d      
#         """        
#         nb,ndata,ndim,nchannel = xc.shape
#         w,b,normalizer,random_w = self.w, self.b, self.normalizer,self.random_w
#         w = w.permute(0,1,3,2)
#         b = b.permute(0,1,3,2)
#         logits_samples = self.logits_samples
     
#         normalizer = np.sqrt(2/self.num_fourierbasis)
#         #normalizer = 1.
#         xc_list = []
#         yc_list = []
        
#         prior_samples_list = []
#         for jch in range(nchannel):
#             j_xc = xc[...,jch]
#             j_xdotw = torch.einsum('nbd,nsdk->nsbk',j_xc,w)

#             if pi2repuse:            
#                 cos_interms = pi2*j_xdotw + pi2*b                   
#             else:
#                 cos_interms = j_xdotw + pi2*b                 

#             Psi = torch.cos(cos_interms)   
#             #print('Psi.shape,random_w.shape')
#             #print(Psi.shape,random_w.shape)
#             #nb,_,ndata,nmixutre = Psi.shape

#             sum_costerm = Psi*random_w[:,:,None,:,jch]            
#             nb,_,ndata,nmixture = sum_costerm.shape
#             sum_costerm = sum_costerm.reshape(nb,numsamples,-1,ndata,nmixture).sum(dim=2)            
#             prior_samples = normalizer*sum_costerm        
#             w_prior_samples = (prior_samples*logits_samples[:,:,jch,None,:]).sum(dim=-1)
            
#             prior_samples_list.append(w_prior_samples)
#             xc_list.append(xc[...,jch])
#             yc_list.append(yc[...,jch])
            
# #        return xc,w,b,random_w       
#         return prior_samples_list,xc_list,yc_list





#     def prepare_updateterms_loop(self,xc,yc,xt,numsamples=1):
#         #prior_shared, xa_shared = self.sample_prior_shared(xc,xt,numsamples=numsamples,reorder=False)
#         prior_ind_list,xc_list,yc_list = self.sample_prior_independent(xc,yc,numsamples=numsamples)


#         mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
#         #logits = self.logits
#         logits = F.softmax(self.loglogits,dim=-1)                
#         likerr = self.loglik.exp()
#         likerr_bound = self.loglik_bound

#         #if yt is None:
#         #xa_shared_ = xa_shared[...,0]    
#         xt_list = [xt[...,j] for j in range(xt.size(-1))]

#         update_term_list = []
#         density_term_list = []
#         for j,(j_xc,j_yc,j_xt,j_prior_ind) in enumerate(zip(xc_list,yc_list,xt_list,prior_ind_list)):
#             param_list_j = (mu,inv_std,logits[j],likerr[j])    
#             #K_cc = eval_smkernel(param_list_j, j_xc, likerr_bound=likerr_bound)
#             #K_tc = eval_smkernel(param_list_j, j_xt , j_xc, likerr_bound=likerr_bound)
            
#             K_cc,exp_term_cc = eval_smkernel(param_list_j, j_xc , likerr_bound=likerr_bound)
#             K_tc,exp_term_tc = eval_smkernel(param_list_j, j_xt , j_xc, likerr_bound=likerr_bound)
#             #print('K_tc.shape {}'.format(K_tc.shape))
            
            
#             L = torch.linalg.cholesky(K_cc)      
#             j_prior_ind = j_prior_ind + likerr[j]*torch.randn_like(j_prior_ind).to(j_xc.device)
#             delta_j_yc = (j_yc[:,None,:] - j_prior_ind).permute(0,2,1)

#             j_Kinvyc = torch.cholesky_solve(delta_j_yc,L,upper=False)
#             j_update_term = torch.einsum('btc,bcs->bts',K_tc,j_Kinvyc).permute(0,2,1)
#             update_term_list.append(j_update_term)
#             #density_term_list.append(exp_term_tc.sum(dim=-1))
#             density_term_list.append(K_tc.sum(dim=-1))

#         return update_term_list,density_term_list,xt_list



#     def sample_posterior_loop(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None):
#         prior_shared_list, xa_shared = self.sample_prior_shared(xc,xt,numsamples=numsamples,reorder=False)
#         prior_ind_list,xc_list,yc_list =  self.sample_prior_independent(xc,yc,numsamples=numsamples)
        
        
#         xa_shared_ = xa_shared[...,None].repeat(1,1,1,self.num_channels)     
#         #update_list,density_term_list,xt_list = self.prepare_updateterms(xc,yc, xa_shared,numsamples=numsamples)
#         update_list,density_term_list,xt_list = self.prepare_updateterms(xc,yc, xa_shared_,numsamples=numsamples)

        
#         posterior_list = []
#         for j,(j_prior,j_update) in enumerate(zip(prior_shared_list,update_list)):
            
#             #print('j_prior.shape,j_update.shape')
#             #print(j_prior.shape,j_update.shape)
            
#             j_posterior = j_prior+j_update
#             posterior_list.append(j_posterior)    
            
#         outs = AttrDict()
#         #outs.xa_samples = xa_shared          #(nb,ndata2,ndim,nchannel)
#         outs.xa_samples = xa_shared    #(nb,ndata2,ndim)                  
#         outs.prior_samples = torch.cat([jth_prior.unsqueeze(dim=-1) for jth_prior in prior_shared_list],dim=-1)       
#         outs.posterior_samples= torch.cat([ jth_posterior.unsqueeze(dim=-1) for jth_posterior in posterior_list],dim=-1)  
#         outs.density = torch.cat([ jth_density.unsqueeze(dim=-1) for jth_density in density_term_list],dim=-1)  
#         outs.regloss = torch.tensor(0.0).float()
        
#         #outs.prior_samples = posterior_list       
#         #outs.prior_samples= torch.gather(outs.prior_samples,2,idx_reorder)                
#         #outs.prior_density = denrep
#         #outs.prior_datarep = datarep        
#         #return  update_term,xa_samples,prior_samples,prior_yc,Kzx
        
#         return outs    



    

#     def prepare_updateterms_loop2(self,xc,yc,xa_shared=None,xt=None,numsamples=1):
#         #xt_list = [xt[...,j] for j in range(xt.size(-1))]
#         nb,ndata,ndim,nchannel = xc.shape


#         mu = self.logmu.exp()
#         inv_std  = 1/(self.logstd.exp()+eps)
#         logits = F.softmax(self.loglogits,dim=-1)        
#         likerr = self.loglik.exp()
#         likerr_bound = self.loglik_bound

#         if xa_shared is not None:
#             xa_shared_ = xa_shared[...,None].repeat(1,1,1,nchannel)
        
#         K_cc_list,K_ac_list = [],[]
#         for j in range(nchannel):    
#             param_list_j = (mu,inv_std,logits[j],likerr[j])    
#             K_cc,exp_term_cc = eval_smkernel(param_list_j, xc[...,j] , likerr_bound=likerr_bound)
#             #K_ac,exp_term_tc = eval_smkernel(param_list_j, xa_shared , xc[...,j] , likerr_bound=likerr_bound)    
#             K_ac,exp_term_tc = eval_smkernel(param_list_j, xa_shared_[...,j] , xc[...,j] , likerr_bound=likerr_bound)    
            
#             K_cc_list.append(K_cc.unsqueeze(dim=1))
#             K_ac_list.append(K_ac.unsqueeze(dim=1))

#         K_cc_ =  torch.cat(K_cc_list,dim=1)
#         K_ac_ =  torch.cat(K_ac_list,dim=1)    
#         #K_cc_.shape,K_ac_.shape
#         L = torch.linalg.cholesky(K_cc_ )  
        
#         w_prior_ind = self.sample_prior_independent(xc,xt,numsamples=numsamples)        
#         w_prior_ind =  w_prior_ind +  likerr[None,None,None,:]*torch.randn_like(w_prior_ind).to(xc.device)
#         delta_yc = yc[:,None,:,:]  - w_prior_ind

#         delta_yc = delta_yc.permute(0,3,2,1)
#         Kinvyc = torch.cholesky_solve(delta_yc,L,upper=False)        
        
#         update_term = torch.einsum('bnac,bncs->bnas',K_ac_,Kinvyc).permute(0,3,2,1)        
#         density_term = K_ac_.sum(dim=-1).permute(0,2,1)    
#         return update_term,density_term 





 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

#-------------------------------------------------------------------------------------------------
#num_mixtures = 3
#num_mixtures = 6 #V47

# class Spikeslab_GPsampler(nn.Module):        
    
#     #def __init__(self,nbasis=10,in_channels=3, nparams = 3, w_mu = None,w_std= None):
# #     def __init__(self,kernel=None,in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 20,num_sampleposterior=10 ,
# #                       scales=.1, loglik_err=1e-2, eps=1e-6,points_per_unit=64,multiplier=2**3):
#     def __init__(self,in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 10,num_sampleposterior=10 ,
#                       scales=.5, loglik_err=1e-2, eps=1e-6,points_per_unit=64,multiplier=2**3):
        
#         super(Spikeslab_GPsampler, self).__init__()
#         #super(Multioutput_GPsampler, self).__init__()
        
#         self.in_dims = in_dims
#         self.out_dims = out_dims
#         self.num_channels = num_channels
#         self.num_fourierbasis = num_fourierbasis
#         #self.kernel = kernel
#         #self.target_idx = self.kernel.target_idx                
#         #self.cross_idx = self.kernel.cross_idx                
        
#         self.normal0 = Normal(loc=0.0,scale=1.0)
#         self.uniform0 = Uniform(0,1)
            
#         self.points_per_unit = points_per_unit  
#         self.multiplier = multiplier        
#         #self.inverseuse = inverseuse
#         self.regloss = 0.0
        
        
# #         # learnable parameters
# #         self.logp =    nn.Parameter( torch.log(eps  +  1.*torch.rand(in_dims,num_channels)) )  #much powerful                                                
# #         self.logmu =    nn.Parameter( torch.log(eps  +  1.*torch.rand(in_dims,num_channels)) )                                               
# #         self.logstd =   nn.Parameter( torch.log(eps + scales*torch.ones(in_dims,num_channels)) )          
# #         self.loglik =   nn.Parameter( torch.log(eps + loglik_err*torch.ones(num_channels)) )
# #         self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
        
#         self.num_mixtures = num_mixtures
#         self.set_initparams(scales=scales,loglik_err = loglik_err,eps =eps)
        
            
#         self.w=None
#         self.b=None        
#         self.normalizer=None
#         self.random_w=None        
#         #self.prior_scale = 0.5        
#         self.prior_scale = 1.0              
#         return 

    
    
#     def set_initparams(self,scales=1.0,loglik_err=1e-2,eps=1e-6):
#         # learnable parameters
#         #logits = eps  +  1.*torch.rand(self.num_channels,self.num_mixtures)
#         #logits = F.softmax(logits,dim=-1)
#         #loglogits = eps  +  1.*torch.rand(self.num_channels,self.num_mixtures)
#         loglogits = eps  +  1.*torch.ones(self.num_channels,self.num_mixtures)
        
        
#         logmu = eps  +  1.*torch.rand(self.num_mixtures,self.in_dims)                
#         logmu[0] = eps*torch.ones(self.in_dims) 
#         #logmu = logmu.sort(dim=0)[0]
#         #logstd = eps + scales*torch.ones(num_channels,in_dims)
#         logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
#         loglik = eps + loglik_err*torch.ones(self.num_channels)
        
#         #self.logits =    nn.Parameter( logits )  #much powerful   
#         self.loglogits =    nn.Parameter( torch.log( loglogits ) )  #much powerful                                                        
        
#         self.logmu =    nn.Parameter( torch.log( logmu )) 
#         self.logstd =   nn.Parameter( torch.log( logstd ))                  
#         self.loglik =   nn.Parameter( torch.log( loglik ))
#         #self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
#         self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
        
#         return 
    
    
    
#     def build_xgrid(self,xc,xt,x_thres=1.0):
#         nb,_,ndim,nchannel=xc.size()         
#         x_min = min(torch.min(xc).cpu().numpy(),torch.min(xt).cpu().numpy()) - x_thres
#         x_max = max(torch.max(xc).cpu().numpy(),torch.max(xt).cpu().numpy()) + x_thres
#         num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))                     
#         xgrid = torch.linspace(x_min,x_max, num_points)
#         xgrid = xgrid.reshape(1,num_points,1).repeat(nb,1,ndim).to(xc.device)
#         return xgrid
        
    

  
#     #def sample_w_b(self,nb,eps=1e-6):    
#     def sample_w_b(self,nb,nsamples,eps=1e-6):    
        
#         """
#         self.w_mu : nparams
#         sample_w  : (nb,nfourifbasis,indim,nchannels)
#         sample_b  : (nb,nfouribasis,indim,nchannels)        
#         """        
#         # (num_mixtures,num_dim)
#         mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        
#         eps1 = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,self.in_dims)).to(mu.device)
#         eps2 = self.uniform0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,1)).to(mu.device)           
#         #random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_channels)).to(mu.device)
#         #random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures)).to(mu.device)
#         random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,self.num_channels)).to(mu.device)

#         sample_w = mu[None,None,:,:] + inv_std[None,None,:,:]*eps1    #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)                
#         sample_b = eps2                                                #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)

#         #return sample_w,sample_b,weight,random_w
#         return sample_w,sample_b,random_w
    
    
    
    
    
    
#     #def sample_prior_shared(self,xc,xt,numsamples=10,reorder=False, temperature= 1e-3):        
#     def sample_prior_shared(self,xc,xt,numsamples=10,reorder=False, temperature= 1e-1):        
        
#         """
#         inputs:
#             xc : (nb,ncontext,ndim,nchannel)
#             xt : (nb,ntarget,ndim,nchannel)        
#         outputs:
#             xa_samples : (nb,nchannel*(ncontext+ntarget),ndim)
#             Psi_pred    : (nb,nchannel*(ncontext+ntarget),nchannel)      # assume y-value is 1-d      
#         """
#         nb = xc.size(0)
#         #if xt in None:
#         #xa_samples = self.samples_xa(xc,xt)                                                 #(nb,nchannel*(ncontext+ntarget),ndim)  
#         xa_samples =self.build_xgrid(xc,xt)
#         #w,b,normalizer,random_w = self.sample_w_b(nb,numsamples)                            #(nb,nbasis,indim,numchannels)    
#         w,b,random_w = self.sample_w_b(nb,numsamples)                            #(nb,nbasis,indim,numchannels)    
        
#         self.w = w
#         self.b = b
#         self.random_w = random_w   
        
#         #(nb,nsamples*nfourier,in_dim,num_mixture)
#         w = w.permute(0,1,3,2)
#         b = b.permute(0,1,3,2)
#         #b_ = b[:,None,:,:]
#         #print(b.shape)
#         xa_samples_ = xa_samples[:,None,:].repeat(1,w.size(1),1,1)
        
#         #print('xa_samples_.shape {}'.format(xa_samples_.shape))
#         #print('w.shape {}'.format(w.shape))
#         #xadotw = torch.einsum('bixy,bjyz->bijxz',xa_samples_,w)
#         xadotw = torch.einsum('bjxy,bjyz->bjxz',xa_samples_,w)
#         #print('xadotw.shape {}'.format(xadotw.shape))
#         #print('b.shape {}'.format(b.shape))
        
#         if pi2repuse:            
#             cos_interms = pi2*xadotw + pi2*b                   
#         else:
#             cos_interms = xadotw + pi2*b                 

#         Psi = torch.cos(cos_interms)            
#         nb,_,ndata2,nmixture = Psi.shape

        
#         Psi = Psi[...,None].repeat(1,1,1,1,self.num_channels)
#         random_w = random_w[:,:,None,:,:]
#         sum_costerm =  (Psi*random_w).reshape(nb,numsamples,self.num_fourierbasis,ndata2,nmixture,self.num_channels)
#         sum_costerm = sum_costerm.sum(dim=2)

#         normalizer = np.sqrt(2/self.num_fourierbasis)        
#         prior_samples = normalizer*sum_costerm        

        
#         logits = F.softmax(self.loglogits,dim=-1)        
#         logits_samples = sample_gumbel_softmax(logits, nb=nb, nsamples=numsamples, temperature= temperature)
#         logits_samples_ = logits_samples.permute(0,1,3,2)[:,:,None,:,:]
#         self.logits_samples = logits_samples
#         w_prior_samples = (prior_samples*logits_samples_).sum(dim=-2)        
#         #print('w_prior_samples.shape')        
#         #print(w_prior_samples.shape)
        
# #         if reorder:
# #             xa_samples,idx_reorder = xa_samples.sort(dim=1)
# #             idx_reorder = idx_reorder[:,None,:,:] 
# #             prior_samples= torch.gather(prior_samples,2,idx_reorder.repeat(1,numsamples,1,nchannel))        
            
#         prior_samples_list = [w_prior_samples[...,j] for j in range(prior_samples.size(-1))]
#         return prior_samples_list,xa_samples             


    
    
    
    #def sample_prior_independent(self,xc,numsamples=10):    
    #def sample_prior_independent(self,xc,numsamples=10):    
    def sample_prior_independent_loop(self,xc,yc,numsamples=10):    
        
        """
        inputs:
            xc : (nb,ncontext,ndim,nchannel)
        outputs:
            Psi_pred : (nb,ncontext,nchannel)      # assume y-value is 1-d      
        """        
        nb,ndata,ndim,nchannel = xc.shape
        w,b,normalizer,random_w = self.w, self.b, self.normalizer,self.random_w
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)
        logits_samples = self.logits_samples
     
        normalizer = np.sqrt(2/self.num_fourierbasis)
        #normalizer = 1.
        xc_list = []
        yc_list = []
        
        prior_samples_list = []
        for jch in range(nchannel):
            j_xc = xc[...,jch]
            j_xdotw = torch.einsum('nbd,nsdk->nsbk',j_xc,w)

            if pi2repuse:            
                cos_interms = pi2*j_xdotw + pi2*b                   
            else:
                cos_interms = j_xdotw + pi2*b                 

            Psi = torch.cos(cos_interms)   
            #print('Psi.shape,random_w.shape')
            #print(Psi.shape,random_w.shape)
            #nb,_,ndata,nmixutre = Psi.shape

            sum_costerm = Psi*random_w[:,:,None,:,jch]            
            nb,_,ndata,nmixture = sum_costerm.shape
            sum_costerm = sum_costerm.reshape(nb,numsamples,-1,ndata,nmixture).sum(dim=2)            
            prior_samples = normalizer*sum_costerm        
            w_prior_samples = (prior_samples*logits_samples[:,:,jch,None,:]).sum(dim=-1)
            
            prior_samples_list.append(w_prior_samples)
            xc_list.append(xc[...,jch])
            yc_list.append(yc[...,jch])
            
#        return xc,w,b,random_w       
        return prior_samples_list,xc_list,yc_list





    def prepare_updateterms_loop(self,xc,yc,xt,numsamples=1):
        #prior_shared, xa_shared = self.sample_prior_shared(xc,xt,numsamples=numsamples,reorder=False)
        prior_ind_list,xc_list,yc_list = self.sample_prior_independent(xc,yc,numsamples=numsamples)


        mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        #logits = self.logits
        logits = F.softmax(self.loglogits,dim=-1)                
        likerr = self.loglik.exp()
        likerr_bound = self.loglik_bound

        #if yt is None:
        #xa_shared_ = xa_shared[...,0]    
        xt_list = [xt[...,j] for j in range(xt.size(-1))]

        update_term_list = []
        density_term_list = []
        for j,(j_xc,j_yc,j_xt,j_prior_ind) in enumerate(zip(xc_list,yc_list,xt_list,prior_ind_list)):
            param_list_j = (mu,inv_std,logits[j],likerr[j])    
            #K_cc = eval_smkernel(param_list_j, j_xc, likerr_bound=likerr_bound)
            #K_tc = eval_smkernel(param_list_j, j_xt , j_xc, likerr_bound=likerr_bound)
            
            K_cc,exp_term_cc = eval_smkernel(param_list_j, j_xc , likerr_bound=likerr_bound)
            K_tc,exp_term_tc = eval_smkernel(param_list_j, j_xt , j_xc, likerr_bound=likerr_bound)
            #print('K_tc.shape {}'.format(K_tc.shape))
            
            
            L = torch.linalg.cholesky(K_cc)      
            j_prior_ind = j_prior_ind + likerr[j]*torch.randn_like(j_prior_ind).to(j_xc.device)
            delta_j_yc = (j_yc[:,None,:] - j_prior_ind).permute(0,2,1)

            j_Kinvyc = torch.cholesky_solve(delta_j_yc,L,upper=False)
            j_update_term = torch.einsum('btc,bcs->bts',K_tc,j_Kinvyc).permute(0,2,1)
            update_term_list.append(j_update_term)
            #density_term_list.append(exp_term_tc.sum(dim=-1))
            density_term_list.append(K_tc.sum(dim=-1))

        return update_term_list,density_term_list,xt_list



    def sample_posterior_loop(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None):
        prior_shared_list, xa_shared = self.sample_prior_shared(xc,xt,numsamples=numsamples,reorder=False)
        prior_ind_list,xc_list,yc_list =  self.sample_prior_independent(xc,yc,numsamples=numsamples)
        
        
        xa_shared_ = xa_shared[...,None].repeat(1,1,1,self.num_channels)     
        #update_list,density_term_list,xt_list = self.prepare_updateterms(xc,yc, xa_shared,numsamples=numsamples)
        update_list,density_term_list,xt_list = self.prepare_updateterms(xc,yc, xa_shared_,numsamples=numsamples)

        
        posterior_list = []
        for j,(j_prior,j_update) in enumerate(zip(prior_shared_list,update_list)):
            
            #print('j_prior.shape,j_update.shape')
            #print(j_prior.shape,j_update.shape)
            
            j_posterior = j_prior+j_update
            posterior_list.append(j_posterior)    
            
        outs = AttrDict()
        #outs.xa_samples = xa_shared          #(nb,ndata2,ndim,nchannel)
        outs.xa_samples = xa_shared    #(nb,ndata2,ndim)                  
        outs.prior_samples = torch.cat([jth_prior.unsqueeze(dim=-1) for jth_prior in prior_shared_list],dim=-1)       
        outs.posterior_samples= torch.cat([ jth_posterior.unsqueeze(dim=-1) for jth_posterior in posterior_list],dim=-1)  
        outs.density = torch.cat([ jth_density.unsqueeze(dim=-1) for jth_density in density_term_list],dim=-1)  
        outs.regloss = torch.tensor(0.0).float()
        
        #outs.prior_samples = posterior_list       
        #outs.prior_samples= torch.gather(outs.prior_samples,2,idx_reorder)                
        #outs.prior_density = denrep
        #outs.prior_datarep = datarep        
        #return  update_term,xa_samples,prior_samples,prior_yc,Kzx
        
        return outs    






 