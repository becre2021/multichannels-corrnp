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


__all__= ['Spikeslab_GPsampler']



class transinvariant_mlp(nn.Module):
    #def __init__(self,in_dims=1,num_channels=1,hdims=10,num_features=5,num_mixtures=1,eps=1e-6):
    def __init__(self,in_dims=1,num_channels=1,hdims=10,num_mixtures=1,eps=1e-6):
        
    #def __init__(self,in_dims=1,hdims=1,num_channels=1,num_features=1,num_mixtures=1,eps=1e-6):
        
        super(transinvariant_mlp,self).__init__()
        
        self.in_dims = in_dims
        self.hdims = hdims
        self.num_channels = num_channels
        self.num_mixtures = num_mixtures

        self.fc1 = nn.Linear(num_mixtures+1,hdims)    
        #self.fc2 = nn.Linear(hdims,hdims)
        self.fc2 = nn.Linear(hdims*num_channels,hdims)    
        self.fc3 = nn.Linear(hdims,hdims)    
        self.fc4 = nn.Linear(hdims,hdims)            
        self.fc5 = nn.Linear(hdims,num_mixtures*num_channels)    

        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)    
        nn.init.xavier_uniform_(self.fc4.weight)    
        nn.init.xavier_uniform_(self.fc5.weight)    


    
    def forward(self,xc,yc,param_list=[]):
        """
        args:
            xc: (nb,ndata,ndim,nchannel)
            yc: (nb,ndata,nchannel)            
        return :
            loglogits: (nb,nmixuture)

        """
        
        nb,ndata,ndim,nchannel=xc.shape        
        
        #(nb,ndata,ndata,nchannel,nmixture)
        Kcc = eval_smkernel_batch(param_list,xc=xc,xt=None,likerr_bound=None)
        #feature_xc = Kcc.sum(dim=2)   #(nb,ndata,nchannel,nmixture)
        feature_xc = (Kcc*yc[:,None,:,:,None]).sum(dim=2)   #(nb,ndata,nchannel,nmixture)        
        transinv_feature = torch.cat([feature_xc,yc.unsqueeze(dim=-1)],dim=-1) #(nb,ndata,nchannel,nmixture+1)                
        h = F.relu(self.fc1(transinv_feature))  #(nb,ndata,nchannel,nmixture+1) 
        h = (h.mean(dim=1)).reshape(nb,-1)    #(nb,num_mixtures*num_channels)                         
        h = F.relu(self.fc2(h)) 
        h = F.relu(self.fc3(h)) 
        h = F.relu(self.fc4(h))         
        loglogits = self.fc5(h)         
        loglogits = loglogits.reshape(nb,self.num_channels,-1)      
        
        #self.param_list = param_list
        return loglogits
        
        
    def compute_feature(self,xc,yc,param_list=[]):
        """
        args:
            xc: (nb,ndata,ndim,nchannel)
            yc: (nb,ndata,nchannel)            
        return :
            loglogits: (nb,nmixuture)

        """
        
        nb,ndata,ndim,nchannel=xc.shape        
        
        #(nb,ndata,ndata,nchannel,nmixture)
        #param_list = self.param_list
        Kcc = eval_smkernel_batch(param_list,xc=xc,xt=None,likerr_bound=None)
        #feature_xc = Kcc.sum(dim=2)   #(nb,ndata,nchannel,nmixture)
        feature_xc = (Kcc*yc[:,None,:,:,None]).sum(dim=2)   #(nb,ndata,nchannel,nmixture)        
        transinv_feature = torch.cat([feature_xc,yc.unsqueeze(dim=-1)],dim=-1) #(nb,ndata,nchannel,nmixture+1)                
        h = F.relu(self.fc1(transinv_feature))  #(nb,ndata,nchannel,nmixture+1) 
        h = (h.mean(dim=1)).reshape(nb,-1)    #(nb,num_mixtures*num_channels)                         
        return h,feature_xc
        
    
    
    
    
    
def sample_gumbel(samples_shape, eps=1e-20):
    #shape = logits.shape
    unif = torch.rand(samples_shape)
    g = -torch.log(-torch.log(unif + eps))
    return g.float()

#def sample_gumbel_softmax(logits, nb=1, nsamples=1, temperature=0.1,training=True):    
def sample_gumbel_softmax(logits, nb=1, nsamples=1, temperature=1.0,training=True):    

#logits_samples = sample_gumbel_softmax(logits, samples_shape, temperature=0.1)    
    """
        Input:
        logits: Tensor of "log" probs, shape = BS x k
        temperature = scalar
        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    if logits.dim() == 2:
#         ## too many randommness
#         nchannel,nmixture = logits.shape
#         g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                
#         logits = logits[None,None,:,:]

        ## reduce randomness
        nchannel,nmixture = logits.shape
        g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)                
        logits = logits[None,None,:,:]

        
        
    if logits.dim() == 3:
        nb,nchannel,nmixture = logits.shape        
        
        if training :
            #g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                             
            #g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)  #v4: not good                                       
            g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                               
        else:
            g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)                             
            
        logits = logits[:,None,:,:]


    h = (g + logits)/temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    #h_max = h.max()
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y




    
    
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
    
    #(nmixutre,ndim),(nmixutre,ndim),(nmixutre),(1)
    mu,inv_std,logits,likerr = param_list         
    #mu_=mu[None,:,None,:]
    #inv_std_=inv_std[None,:,None,:]
    #logits_ = logits[None,:,None,None]
    mu_=mu[None,:,None,:,None]
    inv_std_=inv_std[None,:,None,:,None] 
    
    #(nb,nmixture,ndata,ndim,nchannel)
    xc_ = xc.unsqueeze(dim=1)
    xt_ = xt.unsqueeze(dim=1)
    #xc_.shape,xt_.shape

    exp_xc_ = xc_*inv_std_
    exp_xt_ = xt_*inv_std_
    cos_xc_ = xc_*mu_
    cos_xt_ = xt_*mu_

    exp_term_xc2_ = torch.pow(exp_xc_,2).sum(dim=-2)[:,:,:,None,:] 
    exp_term_xt2_ = torch.pow(exp_xt_,2).sum(dim=-2)[:,:,None,:,:]
    cross_term_ = torch.einsum('bmadk,bmcdk->bmack',exp_xc_,exp_xt_)    
    exp_term = exp_term_xc2_ + exp_term_xt2_ -2*cross_term_    
    cos_term = cos_xc_.sum(dim=-2)[:,:,:,None,:] - cos_xt_.sum(dim=-2)[:,:,None,:,:]

    
    # outs :  #(nb,mixture,ncontext,ntarget,nchannel)     
    outs = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term) if pi2repuse else torch.exp(-0.5*exp_term )*torch.cos(cos_term)
        

        
    if logits is None:
        outs =  outs.permute(0,2,3,4,1)        
        if ndata == ndata2: 
            noise_eye = (zitter)*(torch.eye(ndata)[None,:,:,None,None]).to(xc.device)
            outs = outs + noise_eye     
            return outs #(nb,ncontext,ncontext,nchannel,mixture)  
        else:
            return outs #(nb,ncontext,ntarget,nchannel,mixture)        
            
    else:        
        if logits.dim() == 2:
            #(nchannle,nmixture)         
            logits_ =logits.permute(1,0)[None,:,None,None,:]
            weighted_outs = (outs*logits_).sum(dim=1)                    

        if logits.dim() == 3:
            #(nb,nchannle,nmixture) --> (nb,nmixture,1,1,nchannle)                     
            logits_ = logits.permute(0,2,1)[:,:,None,None,:]
            #(nb,ncontext,ntarget,nchannel)                                     
            weighted_outs = (outs*logits_).sum(dim=1)                
            
        if ndata == ndata2: 
            likerr_ = likerr[None,None,None,:]
            likerr_ = torch.clamp(likerr_,min=likerr_bound[0],max=likerr_bound[1])                
            noise_eye = (zitter+likerr_**2)*(torch.eye(ndata)[None,:,:,None]).to(xc.device)
            weighted_outs = weighted_outs + noise_eye     
                        
            likerr__ = likerr[None,None,None,None,:]
            likerr__ = torch.clamp(likerr__,min=likerr_bound[0],max=likerr_bound[1])                            
            noise_eye2 = (zitter+likerr__**2)*(torch.eye(ndata)[None,None,:,:,None]).to(xc.device)            
            outs = outs + noise_eye2     
            
            return weighted_outs,outs

        else:
            return weighted_outs,outs
            #(nb,ncontext,ntarget,nchannel), (nb,ncontext,ntarget,nchannel,mixture)  
        



        
#class NeuralSpikeslab_GPsampler(nn.Module):        
class Spikeslab_GPsampler(nn.Module):        
   
    def __init__(self,in_dims=1,
                      out_dims=1,
                      num_channels=3, 
                      num_fourierbasis = 10,num_sampleposterior=10 ,
                      scales=.5,
                      loglik_err=1e-2,
                      eps=1e-6,
                      points_per_unit=64,
                      multiplier=2**3,
                      useweightnet = True,hdims=10 ):
        
        #super(NeuralSpikeslab_GPsampler, self).__init__()
        super(Spikeslab_GPsampler, self).__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_channels = num_channels
        self.num_fourierbasis = num_fourierbasis
        
        self.normal0 = Normal(loc=0.0,scale=1.0)
        self.uniform0 = Uniform(0,1)
            
        self.points_per_unit = points_per_unit  
        self.multiplier = multiplier        
        #self.inverseuse = inverseuse
        self.regloss = 0.0
        
        
        
        
        self.useweightnet = useweightnet       
        
        
        if num_channels==1:
            num_mixtures=3                        
            self.weight_net = transinvariant_mlp(in_dims=in_dims,
                                                hdims=hdims,
                                                num_channels=num_channels,
                                                num_mixtures=num_mixtures) 


        if num_channels==2:
            num_mixtures=4                        
            self.weight_net = transinvariant_mlp(in_dims=in_dims,
                                                hdims=hdims,
                                                num_channels=num_channels,
                                                num_mixtures=num_mixtures) 

            
        if num_channels == 3:
            #------------------------------------------------------
            # this setting obatins the improved results for multichannel experiments.
            # this reported results are obatinedb by following settings.
            #------------------------------------------------------
            num_mixtures=5
            self.weight_net = transinvariant_mlp(in_dims=in_dims,
                                                hdims=hdims,
                                                num_channels=num_channels,
                                                num_mixtures=num_mixtures)
            
            # not good restuls,
            #self.weight_net = transinvariant_mlp2(in_dims=in_dims,
            #                                     num_channels=num_channels,                                                 
            #                                     hdims=hdims,
            #                                     num_mixtures=num_mixtures ) 

            
            #------------------------------------------------------
            # for illustration figure
            #------------------------------------------------------            
#             num_mixtures=2
#             self.weight_net = transinvariant_mlp(in_dims=in_dims,
#                                                 hdims=hdims,
#                                                 num_channels=num_channels,
#                                                 num_mixtures=num_mixtures)
            
        self.num_mixtures = num_mixtures
        self.set_initparams(scales=scales,loglik_err = loglik_err,eps =eps)
        
          
            
        
        self.w=None
        self.b=None        
        self.normalizer=None
        self.random_w=None        
        #self.prior_scale = 0.5        
        self.prior_scale = 1.0              
        
        self.use_constrainedprior = False #True means density aware prior 
        #self.use_constrainedprior = True #True means density aware prior 
        #self.tempering0 = 0.1
        
        #self.temperature= 1e-3        
        #self.tempering0 = 1.0
        self.tempering0 = 1e-1
        #self.tempering0 = 10.0
        
        
        print('spikeslab version 7 with tempering {}'.format(self.tempering0))
        self.param_list = None
        
        return 

    

    def bound_hypparams(self,eps=1e-6):
        """
        bound_std = [1.,2.] --> bound_invstd = [.5,1.] 
        """        
        bound_logmu = np.log(self.bound_mu+eps)
        bound_logstd = np.log(self.bound_std+eps)        
        #print(self.bound_std )
        
        with torch.no_grad():
            self.logmu.data.clip_(bound_logmu[0],bound_logmu[1])            
            self.logstd.data.clip_(bound_logstd[0],bound_logstd[1])            
        return

    
    
    
    def set_initparams(self,scales=1.0,loglik_err=1e-2,eps=1e-6):
        # learnable parameters
        #loglogits = eps  +  1.*torch.rand(self.num_channels,self.num_mixtures)
        loglogits = eps  +  1.*torch.ones(self.num_channels,self.num_mixtures) + .1*torch.rand(self.num_channels,self.num_mixtures)

        if self.num_channels == 1:                    
            maxfreq = 5
            centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)   
            logmu[0] = eps*torch.ones(self.in_dims)             
            logmu = logmu.sort(dim=0)[0]        
            
            logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
            
            
            #0.25*(maxfreq/self.num_mixtures) >= invstd           
            #4*(self.num_mixtures/maxfreq) <= std
            
            self.bound_mu = np.array([eps,maxfreq])
            #self.bound_std = [1.,10.]   #--> invstd = [0.1,1]
            #self.bound_std = [2.,10.]   #--> invstd = [0.1,1]
            #self.bound_std = np.array([2*(self.num_mixtures/maxfreq),10])
            self.bound_std = np.array([1,5])

            
        if self.num_channels == 2:                    
            maxfreq = 5
            centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)   
            logmu[0] = eps*torch.ones(self.in_dims)             
            logmu = logmu.sort(dim=0)[0]        
            
            logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
            
            
            #0.25*(maxfreq/self.num_mixtures) >= invstd           
            #4*(self.num_mixtures/maxfreq) <= std
            
            self.bound_mu = np.array([eps,maxfreq])
            #self.bound_std = [1.,10.]   #--> invstd = [0.1,1]
            #self.bound_std = [2.,10.]   #--> invstd = [0.1,1]
            #self.bound_std = np.array([2*(self.num_mixtures/maxfreq),10])
            self.bound_std = np.array([1,5])
            

        #-----------------------------
        #settings for reported results
        #-----------------------------                
        if self.num_channels == 3:        
            maxfreq = 5
            centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)                        
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)              
            logmu[0] = eps*torch.ones(self.in_dims) 
            logmu = logmu.sort(dim=0)[0]                                
            
            logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
        
            self.bound_mu = np.array([eps,maxfreq])
            #self.bound_std = [1.,10.]    #--> invstd = [0.1,1]
            #self.bound_std = [2.,10.]    #--> invstd = [0.1,1]
            #self.bound_std = np.array([2*(self.num_mixtures/maxfreq),10])
            self.bound_std = np.array([1,5])



#         # -----------------------------
#         # for illustration figure
#         # -----------------------------        
#         if self.num_channels == 3:        
#             maxfreq = 4
#             centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)                        
#             logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)              
#             logmu[0] = eps*torch.ones(self.in_dims) 
#             logmu = logmu.sort(dim=0)[0]                                
            
#             logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
        
#             self.bound_mu = np.array([eps,maxfreq])
#             #self.bound_std = [1.,10.]    #--> invstd = [0.1,1]
#             #self.bound_std = [2.,10.]    #--> invstd = [0.1,1]
#             #self.bound_std = np.array([2*(self.num_mixtures/maxfreq),10])
#             self.bound_std = np.array([1,5])

                        
        loglik = eps + loglik_err*torch.ones(self.num_channels)
        self.loglogits =    nn.Parameter( torch.log( loglogits ) )  #much powerful                                                        
        self.logmu =    nn.Parameter( torch.log( logmu )) 
        self.logstd =   nn.Parameter( torch.log( logstd ))                   
        #self.logmu =    nn.Parameter( torch.log( logmu )   ,requires_grad=False)         
        #self.logstd =   nn.Parameter( torch.log( logstd )  ,requires_grad=False)                          
        self.loglik =   nn.Parameter( torch.log( loglik ))
        self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
        return 
    
    
    
    
    def build_xgrid(self,xc,xt,x_thres=1.0):
        nb,_,ndim,nchannel=xc.size()         
        x_min = min(torch.min(xc).cpu().numpy(),torch.min(xt).cpu().numpy()) - x_thres
        x_max = max(torch.max(xc).cpu().numpy(),torch.max(xt).cpu().numpy()) + x_thres
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))                     
        xgrid = torch.linspace(x_min,x_max, num_points)
        xgrid = xgrid.reshape(1,num_points,1).repeat(nb,1,ndim).to(xc.device)
        return xgrid
        
    

    def sample_w_b(self,xc,yc,nsamples,eps=1e-6):    
       
        """
        self.w_mu : nparams
        sample_w  : (nb,nfourifbasis,indim,nchannels)
        sample_b  : (nb,nfouribasis,indim,nchannels)        
        """        
        nb = xc.size(0)
        
        # (num_mixtures,num_dim)
        mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        
        eps1 = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,self.in_dims)).to(mu.device)
        eps2 = self.uniform0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,1)).to(mu.device)           
        
        
        #print('allow channel dependency')
        random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,1)).to(mu.device) #impose depedency over channels
        sample_w = mu[None,None,:,:] + inv_std[None,None,:,:]*eps1    #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)                
        sample_b = eps2                                                #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)

        #return sample_w,sample_b,weight,random_w
        #return sample_w,sample_b,random_w,logits_samples
        self.w = sample_w
        self.b = sample_b
        self.random_w = random_w           
        return sample_w,sample_b,random_w

    
    
    def sample_logits(self,xc,yc,nsamples,eps=1e-6,tempering0=1e1):        
        nb = xc.size(0)

        if self.useweightnet:
            mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
            
            #dictionary kernel are used for computing Kcc().sum(dim=2) without learnable kernel parameter learning
            param_list = (mu.detach().clone(),inv_std.detach().clone(),None,None)
            loglogits = self.weight_net(xc,yc,param_list=param_list)
            
            if self.tempering0 is None:
                self.tempering0 = tempering0
            self.neural_loglogits=loglogits / self.tempering0
            logits = F.softmax(self.neural_loglogits  , dim=-1)    #(nb,nchannle,nmixture)   
            self.neural_logits= logits 
            #logits_samples = sample_gumbel_softmax(logits,
            #                                       nb=nb,
            #                                       nsamples=nsamples, 
            #                                       temperature= self.temperature,
            #                                       training = self.training)
            logits_samples = sample_gumbel_softmax(self.neural_loglogits,
                                                   nb=nb,
                                                   nsamples=nsamples, 
                                                   temperature= 1.,
                                                   training = self.training)
            

        else:
            logits = self.loglogits / self.tempering0      #(num_channels,num_mixtures)             
            logits_samples = sample_gumbel_softmax(logits,
                                                   nb=nb,
                                                   nsamples=nsamples,
                                                   temperature= 1.,
                                                   training = self.training)

            
        self.logits_samples= logits_samples
        return logits_samples
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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

        # -----------------------------------
        # prepare cosine features
        # -----------------------------------        
        xa_samples =self.build_xgrid(xc,xt)
        w,b,random_w = self.sample_w_b(xc,yc,numsamples)    
        
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)

        xa_samples_ = xa_samples[:,None,:].repeat(1,w.size(1),1,1)
        xadotw = torch.einsum('bjxy,bjyz->bjxz',xa_samples_,w)        
        cos_interms = pi2*xadotw + pi2*b  if pi2repuse else  xadotw + pi2*b                  
        Psi = torch.cos(cos_interms)            
        nb,_,ndata2,nmixture = Psi.shape

        
        Psi = Psi[...,None].repeat(1,1,1,1,self.num_channels)
        random_w = random_w[:,:,None,:,:]
        sum_costerm =  (Psi*random_w).reshape(nb,numsamples,self.num_fourierbasis,ndata2,nmixture,self.num_channels)
        sum_costerm = sum_costerm.sum(dim=2)
        normalizer = np.sqrt(2/self.num_fourierbasis)        
        prior_samples = normalizer*sum_costerm        #(nb,numsamples,ndata2,nmixture,nchannel)

        
        
        
        
        # -----------------------------------
        # prepare logits features
        # -----------------------------------                
        logits_samples = self.sample_logits(xc,yc,numsamples,eps=1e-6)
        
        #logits_samples : (nb,numsamples,nchannel,nmixture) --> (nb,numsamples,1,nmixture,nchannel)         
        logits_samples_ = logits_samples.permute(0,1,3,2)[:,:,None,:,:]
        w_prior_samples = (prior_samples*logits_samples_).sum(dim=-2)                
        #return w_prior_samples,xa_samples                 
        return w_prior_samples,xa_samples,prior_samples                 


 

    #def sample_prior_independent(self,xc,yc,numsamples=10,newsample=False):            
    def sample_prior_independent(self,xc,numsamples=10,newsample=False):            
        
        nb,ndata,ndim,nchannel = xc.shape
        w,b,random_w,logits_samples = self.w, self.b,self.random_w,self.logits_samples 
            
            
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)
            
        xc_ = xc[:,None,:,:,:].repeat(1,w.size(1),1,1,1)
        w_ =w[...,None].repeat(1,1,1,1,nchannel)
        b_ =b[...,None].repeat(1,1,1,1,nchannel)

        if pi2repuse:            
            xcdotw_b = pi2*(torch.einsum('bsndc,bsdmc->bsnmc',xc_,w_) + b_)
        else:
            xcdotw_b = torch.einsum('bsndc,bsdmc->bsnmc',xc_,w_) + pi2*b_
        
        Psi = torch.cos(xcdotw_b)
        sum_costerm = Psi*random_w[:,:,None,:,:]
        sum_costerm_ = sum_costerm.reshape(nb,numsamples,-1,ndata,self.num_mixtures,nchannel)
        normalizer = np.sqrt(2/self.num_fourierbasis)
        prior_samples = (sum_costerm_.sum(dim=2))*normalizer  #(nb,numsamples,ndata,nmixture,nchannel)
        
        
        #logits_samples : (nb,numsamples,nchannel,nmixture) --> (nb,numsamples,1,nmixture,nchannel) 
        logits_samples_ = logits_samples.permute(0,1,3,2)[:,:,None,:,:]
        w_prior_samples = (prior_samples*logits_samples_).sum(dim=-2)                                    
        
        #(nb,numsamples,ndata,nchannel),(nb,numsamples,ndata,nmixture,nchannel)         
        return w_prior_samples,prior_samples



    
    
    
    
    def prepare_updateterms(self,xc,yc,xa_shared=None,xt=None,numsamples=1):
        nb,ndata,ndim,nchannel = xc.shape

        likerr = self.loglik.exp()
        likerr_bound = self.loglik_bound

        #mu = self.logmu.exp()
        #inv_std  = 1/(self.logstd.exp()+eps)
        #logits = F.softmax(self.loglogits,dim=-1)        
        #logits = F.softmax(self.neural_loglogits,dim=-1)                
        
        mu = self.logmu.exp()
        inv_std  = 1/(self.logstd.exp()+eps)        
        logits = self.neural_logits 

        
        param_list = (mu,inv_std,logits,likerr)
        self.param_list = param_list
        xa_shared_ = xa_shared[...,None].repeat(1,1,1,nchannel)
        WK_cc,K_cc = eval_smkernel_batch(param_list,xc,  likerr_bound=likerr_bound)                
        WK_ac,_ = eval_smkernel_batch(param_list,xa_shared_, xc,  likerr_bound=likerr_bound)
        WK_cc_ = WK_cc.permute(0,3,1,2) 
        WK_ac_ = WK_ac.permute(0,3,1,2) #(nb,nchannel,ndata2,ndata)
        
        
        L = torch.linalg.cholesky(WK_cc_ )                  
        #(nb,numsamples,ndata,nchannel),(nb,numsamples,ndata,nmixture,nchannel)         
        w_prior_ind,prior_ind = self.sample_prior_independent(xc,numsamples=numsamples)                
        w_prior_ind =  w_prior_ind +  likerr[None,None,None,:]*torch.randn_like(w_prior_ind).to(xc.device)
        density_term = WK_ac_.sum(dim=-1).permute(0,2,1)    
                
        delta_yc = yc[:,None,:,:]  - w_prior_ind
        delta_yc = delta_yc.permute(0,3,2,1)
        Kinvyc = torch.cholesky_solve(delta_yc,L,upper=False)        
        update_term_shared = torch.einsum('bnac,bncs->bnas',WK_ac_,Kinvyc).permute(0,3,2,1)        
        
        
        
        
        #--------------------------------------------
        # aim at computing the posterior of gp sapmling
        #--------------------------------------------
        
        #K_tc = eval_smkernel_batch(param_list, xt.clone(), xc.clone(),  likerr_bound=likerr_bound) 
        #K_tc_ = K_tc.permute(0,3,1,2)        
        #update_term_target = torch.einsum('bnac,bncs->bnas',K_tc_,Kinvyc).permute(0,3,2,1)        

        
        #--------------------------------------------
        # computing individual posterior sampling 
        #--------------------------------------------        
        #K_tc : (nb,)
        #(nb,ncontext,ntarget,nchannel), (nb,ncontext,ntarget,nchannel,mixture)  
        
         #(nb,mixture,ncontext,ntarget,nchannel)
        _,K_tc = eval_smkernel_batch(param_list, xt.clone(), xc.clone(),  likerr_bound=likerr_bound) 
        #K_tc_ = K_tc.permute(0,3,1,2)
        #K_tc_ = K_tc.permute(0,1,4,2,3)
        
    
    
        #K_cc.shape,K_tc.shape,prior_ind.shape,yc.shape
        #torch.Size([4, 5, 20, 20, 3]) torch.Size([4, 5, 40, 20, 3]) torch.Size([4, 5, 20, 5, 3]) torch.Size([4, 20, 3])
        #print('K_cc.shape,K_tc.shape,prior_ind.shape,yc.shape')
        #print(K_cc.shape,K_tc.shape,prior_ind.shape,yc.shape)
    
        prior_ind =  prior_ind +  likerr[None,None,None,None,:]*torch.randn_like(prior_ind).to(xc.device)    
        delta_yc2 =  yc[:,None,:,None,:]  - prior_ind
        delta_yc2 = delta_yc2.permute(0,3,4,2,1) #(nb,nmixture,nchannel,ndata,nsamples)    
        K_tc2 = K_tc.permute(0,1,4,2,3)    #(nb,nmixture,nchannel,ndata,nsamples)  
        K_cc2 = K_cc.permute(0,1,4,2,3)   #(nb,nmixture,nchannel,ndata,nsamples)  
    
        L2 = torch.linalg.cholesky(K_cc2)   
        Kinvyc2 = torch.cholesky_solve(delta_yc2,L2,upper=False)        
        update_term_target = torch.einsum('bmntc,bmncs->bmnts',K_tc2,Kinvyc2)  
        update_term_target = update_term_target.permute(0,4,3,1,2) #(nb,nsamples,ndata,nmixture,nchannel)
        #print('update_term_target.shape')
        #print(update_term_target.shape)        
        #update_term_target.shape
        #torch.Size([4, 5, 3, 20, 5])
        
    
    
        
        #return update_term_shared,density_term,update_term_target 
        return update_term_shared,density_term,update_term_target 
    
    
    

    #def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None,use_constrainedprior=False):
    def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None,use_constrainedprior=True):
        
        #prior_shared, xa_shared = self.sample_prior_shared(xc,xt,numsamples=numsamples)
        w_prior_shared, xa_shared , prior_shared = self.sample_prior_shared(xc,yc,xt,numsamples=numsamples)                
        w_update_term_shared, density_term, update_term_target = self.prepare_updateterms(xc,yc,
                                                                                          xa_shared=xa_shared,
                                                                                          xt=xt,numsamples=numsamples)               
        posterior_shared =  w_prior_shared + w_update_term_shared
                
        #if self.use_constrainedprior:        
        #    allow_prior = 1. - 2*(torch.sigmoid(density_term/0.1)-.5) + eps #(nb,ndata2,nchannels)
        #    allow_prior = allow_prior[:,None,:,:].detach().clone()
        #    prior_shared  = prior_shared*allow_prior

    
            
        #(nb,numsamples,ndata,nchannel),(nb,numsamples,ndata,nmixture,nchannel)                         
        w_prior_target ,prior_target = self.sample_prior_independent(xt,numsamples=numsamples)               
        
        #update_term_target.shape
        #torch.Size([4, 5, 3, 20, 5])
        #posterior_target = prior_target + update_term_target[:,:,:,None,:]
        
        #revision for individual prior
        posterior_target = prior_target + update_term_target
        
        outs = AttrDict()
        outs.xa_samples = xa_shared    #(nb,ndata2,ndim)                  
        outs.prior_samples =  prior_shared      
        outs.wprior_samples = w_prior_shared
        outs.posterior_samples= posterior_shared         
        outs.posterior_target= posterior_target 
        outs.neural_loglogits = self.neural_loglogits
        outs.neural_logits = self.neural_logits
        outs.tempering0 = self.tempering0
        
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






 














# class deepset_mlp(nn.Module):
#     def __init__(self,in_dims=1,hdims=1,num_channels=1,num_mixtures=1):
#         super(deepset_mlp,self).__init__()
        
#         self.in_dims = in_dims
#         self.hdims = hdims
#         self.num_channels = num_channels
#         self.num_mixtures = num_mixtures
        
        
#         self.fc1 = nn.Linear(in_dims,hdims)
#         self.fc2 = nn.Linear(hdims,hdims)
#         self.fc3 = nn.Linear(hdims*num_channels,num_mixtures*num_channels)    
# #         self.fc1 = nn.Linear(in_dims,hdims , bias=False)
# #         self.fc2 = nn.Linear(hdims,hdims , bias=False)
# #         self.fc3 = nn.Linear(hdims*num_channels,num_mixtures*num_channels , bias=False)     
    
#         #nn.init.xavier_uniform(self__.fc1.weight)   
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)
    
    
#     def forward(self,xc,yc):
#         """
#         args:
#             xc: (nb,ndata,ndim,nchannel)
#             yc: (nb,ndata,nchannel)
            
#         return :
#             loglogits: (nb,nmixuture)

#         """
#         nb,ndata,ndim,nchannel=xc.shape
#         xc = xc.permute(0,3,1,2)   
#         yc = yc.permute(0,2,1)        
        
        
#         h = F.relu(self.fc1(xc))
#         h = self.fc2(h)
#         h = (h*yc[...,None]).mean(dim=-2)
#         h = h.reshape(nb,-1)        
        
#         loglogits= self.fc3(F.relu(h)) 
#         loglogits = loglogits.reshape(nb,self.num_channels,-1)
#         return loglogits




# def sample_gumbel_softmax(logits, nb=1, nsamples=1, temperature=0.1):    
# #logits_samples = sample_gumbel_softmax(logits, samples_shape, temperature=0.1)    
#     """
#         Input:
#         logits: Tensor of log probs, shape = BS x k
#         temperature = scalar
        
#         Output: Tensor of values sampled from Gumbel softmax.
#                 These will tend towards a one-hot representation in the limit of temp -> 0
#                 shape = BS x k
#     """
#     if logits.dim() == 2:
# #         ## too many randommness
# #         nchannel,nmixture = logits.shape
# #         g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                
# #         logits = logits[None,None,:,:]

#         ## reduce randomness
#         nchannel,nmixture = logits.shape
#         g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)                
#         logits = logits[None,None,:,:]

#     if logits.dim() == 3:

# #         ## original intention, 
# #         ## maek too many randommness, which makes NN difficult to be trained
# #         nb,nchannel,nmixture = logits.shape
# #         g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                
# #         logits = logits[:,None,:,:]

#         nb,nchannel,nmixture = logits.shape        
#         if nchannel ==1:
#             #print("allow more randomness, which is original way")                                            
#             g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                             
            
#             #print("allow channel dependency")                                
#             #g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)                             
            
#         if nchannel ==3:
#             #print("allow channel dependency")                    
#             g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)                             
            
#         logits = logits[:,None,:,:]


#     h = (g + logits)/temperature
#     h_max = h.max(dim=-1, keepdim=True)[0]
#     #h_max = h.max()
#     h = h - h_max
#     cache = torch.exp(h)
#     y = cache / cache.sum(dim=-1, keepdim=True)
#     return y






#     def compute_Kcc(self,xc, eps=1e-6):
#         """
#         args:
#             xc : (nb,ndata,ndim,nchannel)
#         return:
#             Kcc : (nb,ndata,ndata,nchannel,nfeatures)          
#         """        
#         assert xc.dim() == 4        
#         xc_ = xc[:,:,:,:,None]
#         length_scale = self.length_scale.exp() + eps
#         length_scale_ = length_scale[None,None,:,:,:]         
#         #print(xc_.shape,length_scale_.shape)
        
#         xc_ = xc_/length_scale_         
#         xc2_ = (xc_**2).sum(dim=-3)
#         xc2_square = xc2_[:,:,None,:,:] + xc2_[:,None,:,:,:]
#         xc_cross= torch.einsum('bndcf,bmdcf->bnmcf',xc_,xc_)
#         dist = xc2_square -2*xc_cross
#         Kcc = torch.exp(-0.5*dist)
#         return Kcc #(nb,ndata,ndata,nchannel,nfeatures) 
        
        
#     def transinv_feature(self,xc,yc):
#         """
#         args:
#             xc: (nb,ndata,ndim,nchannel)
#             yc: (nb,ndata,nchannel)            
#         return :
#             loglogits: (nb,nmixuture)

#         """
#         nb,ndata,ndim,nchannel=xc.shape
#         Kcc = self.compute_Kcc(xc)
#         feature_xc = Kcc.sum(dim=2)    #torch.Size([16, 50, 50, 3 ,5 ]) ->  torch.Size([16, 50, 3, 5])    

#         #torch.Size([16, 50, 3, 5+1])  or         #torch.Size([16, 50, 1, 5+1])                
#         transinv_feature = torch.cat([feature_xc,yc.unsqueeze(dim=-1)],dim=-1)
#         return transinv_feature        
        
    

# def eval_smkernel(param_list,xc,xt=None, likerr_bound=[.1,1.0], zitter=1e-4):
# #def eval_smkernel(param_list,xc,xt=None, likerr_bound=[.1,1.0], zitter=1e-3):    
# #def eval_smkernel(param_list,xc,xt=None, likerr_bound=[.1,1.0], zitter=1e-3,zitter_flag = False):

#     """
#     inputs:
#         #xc : (nb,ncontext,ndim,nchannel)
#         xc : (nb,ncontext,ndim)
        
#     outputs:
#         Kxz : (nb,ncontext,ncontext,nchannel)      # assume y-value is 1-d      
#     """
    
#     assert len(xc.shape) == 3                            
#     if xt is None:
#         xt = xc
#         nb,ndata,ndim = xc.size()                                   
#     else:
#         nb,ndata,ndim = xc.size()                            
#         _,ndata2,_ = xt.size()                            

#     xc_ = xc.unsqueeze(dim=1)
#     xt_ = xt.unsqueeze(dim=1)
        
        
#     assert len(param_list) == 4    
#     #(nmixutre,ndim),(nmixutre,ndim),(nmixutre),(1)
#     mu,inv_std,logits,likerr = param_list         
#     mu_=mu[None,:,None,:]
#     inv_std_=inv_std[None,:,None,:]
#     logits_ = logits[None,:,None,None]
#     #print('mu.shape,inv_std.shape,logits.shape,likerr.shape')
#     #print(mu.shape,inv_std.shape,logits.shape,likerr.shape)


#     exp_xc_ = xc_*inv_std_
#     exp_xt_ = xt_*inv_std_
#     cos_xc_ = xc_*mu_
#     cos_xt_ = xt_*mu_

#     exp_term_xc2_ = torch.pow(exp_xc_,2).sum(dim=-1)[:,:,:,None] 
#     exp_term_xt2_ = torch.pow(exp_xt_,2).sum(dim=-1)[:,:,None,:]
#     cross_term_ = torch.einsum('bmad,bmcd->bmac',exp_xc_,exp_xt_)    
#     exp_term = exp_term_xc2_ + exp_term_xt2_ -2*cross_term_    
#     cos_term = cos_xc_.sum(dim=-1)[:,:,:,None] - cos_xt_.sum(dim=-1)[:,:,None,:]
    
#     if pi2repuse:    
#         #outs = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
#         exp_term_,cos_term_ = torch.exp(-0.5*(pi2**2)*exp_term ),torch.cos(pi2*cos_term)        
#     else:
#         #outs = torch.exp(-0.5*exp_term )*torch.cos(cos_term)
#         exp_term_,cos_term_ = torch.exp(-0.5*exp_term),torch.cos(cos_term)

#     outs = exp_term_*cos_term_     
#     weighted_outs = (outs*logits_).sum(dim=1)    
#     weighted_exp_term = (exp_term_*logits_).sum(dim=1)    

    
#     if ndata == ndata2:
#         likerr = torch.clamp(likerr,min=likerr_bound[0],max=likerr_bound[1])               
#         noise_eye = (zitter+likerr**2)*(torch.eye(ndata)[None,:,:]).to(xc.device)
#         weighted_outs = weighted_outs + noise_eye     
        
#         return weighted_outs, weighted_exp_term 
#     else:
#         return weighted_outs, weighted_exp_term 
    
        