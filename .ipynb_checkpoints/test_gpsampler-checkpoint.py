from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import torch
import numpy as np

import torch.nn as nn
import math
import itertools as it

#from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.utils import to_multiple
from test_kernels import pi2repuse




__all__= ['Independent_GPsampler','Multioutput_GPsampler']


#inverseuse=False    
inverseuse=True
num_foureirbasis=20
pi2 = 2*math.pi
eps=1e-6

#class Multioutput_GPsampler(nn.Module):    
class Independent_GPsampler(nn.Module):        
    #def __init__(self,nbasis=10,in_channels=3, nparams = 3, w_mu = None,w_std= None):
    def __init__(self,kernel=None,in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 20,num_sampleposterior=10 ,
                      scales=.1, loglik_err=1e-2, eps=1e-6,points_per_unit=64,multiplier=2**3):
        super(Independent_GPsampler, self).__init__()
        #super(Multioutput_GPsampler, self).__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_channels = num_channels
        self.num_fourierbasis = num_fourierbasis
        self.kernel = kernel
        self.target_idx = self.kernel.target_idx                
        self.normal0 = Normal(loc=0.0,scale=1.0)
        self.uniform0 = Uniform(0,1)
            
        self.points_per_unit = points_per_unit  
        self.multiplier = multiplier
        
        self.inverseuse = inverseuse
        self.regloss = 0.0
        
            
        self.w=None
        self.b=None        
        self.normalizer=None
        self.random_w=None
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
    def sample_w_b(self,nb,nsamples,eps=1e-6):    
        
        """
        self.w_mu : nparams
        sample_w  : (nb,nfourifbasis,indim,nchannels)
        sample_b  : (nb,nfouribasis,indim,nchannels)        
        """        
        mu,inv_std,weight = self.kernel.prepare_cross_params() #(indims,nchannels,nchannels),(indims,nchannels,nchannels)
        mu = mu[...,self.target_idx]
        inv_std = inv_std[...,self.target_idx]
        weight = weight[...,self.target_idx]
        
        eps1 = self.normal0.sample((nb,self.num_fourierbasis,self.in_dims,self.num_channels)).to(mu.device)
        eps2 = self.uniform0.sample((nb,self.num_fourierbasis,self.in_dims,self.num_channels)).to(mu.device)    
        random_w = self.normal0.sample((nb,nsamples,self.num_fourierbasis,self.num_channels)).to(mu.device)
                
        #sample_w = mu_[None,None,:,:] + std_[None,None,:,:]*eps1    #(nb,nfouierbasis,indims,nchannels,nchannels)         
        sample_w = mu[None,None,:,:] + inv_std[None,None,:,:]*eps1    #(nb,nfouierbasis,indims,nchannels,nchannels)                 
        sample_b = eps2                                                #(nb,nfouierbasis,indims,nchannels,nchannels)
        return sample_w,sample_b,weight,random_w

    
    def sample_prior_shared(self,xc,xt,numsamples=10,reorder=False):        
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
        w,b,normalizer,random_w = self.sample_w_b(nb,numsamples)                            #(nb,nbasis,indim,numchannels)    
        self.w = w
        self.b = b
        self.normalizer =normalizer
        self.random_w = random_w        
        
        # inner product : cos in terms
        xa_samples_ = xa_samples[:,None,:,:,None].repeat(1,1,1,1,self.num_channels)
        w = w[:,:,None,:,:]                        
        
        #(nb,1,nchannel*(ncontext+ntarget),ndim,1)*(nb,nfourierbasis,1,ndim,numchannels)         
        if pi2repuse:
            cos_interms = pi2*(xa_samples_*w).sum(dim=-2) + pi2*b                   
        else:
            cos_interms = (xa_samples_*w).sum(dim=-2) + pi2*b         
        
        Psi = torch.cos(cos_interms).unsqueeze(dim=1)                                        #(nb,nfourierbasis,nchannel*(ncontext+ntarget),nunchannels)
        Psi = Psi.repeat(1,numsamples,1,1,1)                                                 #(nb,numsamples,nfourierbasis,nchannel*(ncontext+ntarget),nunchannels)
        Psi = Psi.permute(0,1,3,2,4) 
        nb,_,ndata,_,_ =Psi.size()        
        
        
        #scale_sigma = self.logsigma.exp().reshape(1,1,1,-1)                
        #prior_samples = scale_sigma*np.sqrt(2/self.num_fourierbasis)*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w)        
        normalizer = normalizer[None,None,:,:].sqrt()
        prior_samples = normalizer*np.sqrt(2/self.num_fourierbasis)*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w)        
                    
        if reorder:
            xa_samples,idx_reorder = xa_samples.sort(dim=1)
            idx_reorder = idx_reorder[:,None,:,:] 
            prior_samples= torch.gather(prior_samples,2,idx_reorder.repeat(1,numsamples,1,nchannel))        
        return xa_samples,prior_samples     

                
    
    
    def sample_prior_independent(self,xc,numsamples=10):    
        """
        inputs:
            xc : (nb,ncontext,ndim,nchannel)
        outputs:
            Psi_pred : (nb,ncontext,nchannel)      # assume y-value is 1-d      
        """        
        nb = xc.size(0)
        #w,b = self.sample_w_b(nb)                                                           #(nb,nbasis,indim,numchannels)     
        #w,b = self.w[...,self.target_idx],self.b[...,self.target_idx]      
        w,b,normalizer,random_w = self.w, self.b, self.normalizer,self.random_w
        
        xc_ = xc[:,None,:,:,:] #(nb,1,ndata,ndim,nchannels)
        w_ =   w[:,:,None,:,:]         
        
        if pi2repuse:        
            cos_interms = pi2*(xc_*w_).sum(dim=-2) +pi2*b  
        else:
            cos_interms = (xc_*w_).sum(dim=-2) +pi2*b  
        
        Psi = torch.cos(cos_interms) 
        Psi = Psi.unsqueeze(dim=1)
        Psi = Psi.repeat(1,numsamples,1,1,1)            #(nb,numsamples,nfourierbasis,ncontext,nchannels)                
        Psi = Psi.permute(0,1,3,2,4)
        _,numsample,ndata,nfbasis,nchannels = Psi.size()
        
        #scale_sigma = self.logsigma.exp().reshape(1,1,1,-1)        
        #prior_samples = scale_sigma*np.sqrt(2/self.num_fourierbasis)*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w) 
        
        normalizer = normalizer[None,None,:,:].sqrt()*np.sqrt(2/self.num_fourierbasis)
        prior_yc = normalizer*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w)        
        
        #Psi_pred = np.sqrt(2/self.num_fourierbasis)*(Psi*random_w).sum(dim=2)      #(nb,numsamples,ngrid,numchannels)               
        return xc,prior_yc    

    
    
    def prepare_updateterms(self,xc,yc,numsamples=1):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
        outputs
            L: #(nb,nchannels,ndata,ndata)
            delta_yc: #(nb,,nchannels,ndata,nsamples)            
        """
        nb,ndata,ndim,nchannel=xc.size()
        nb,ndata,nchannel=yc.size()
        xc,prior_yc = self.sample_prior_independent(xc,numsamples=numsamples)
        likerr = (self.kernel.loglik).exp()[None,None,None,:] 
        delta_yc = yc[:,None,:,:] - (prior_yc  + likerr*torch.randn(1,numsamples,1,nchannel).to(xc.device))         #(nb,numsample,ndata,nchannels)        
        delta_yc= delta_yc.permute(0,3,2,1)                           #(nb,nchannels,ndata,numsamples)  

        
        Kxx = self.kernel.eval_Kxx(xc)
        Kxx = Kxx.permute(0,3,1,2)              #(nb,,nchannels,ndata,data)  
        L = torch.linalg.cholesky(Kxx)                                                                             #(nb,nchannels,ndata,ndata) 
        
        Kinv_yc = torch.cholesky_solve(delta_yc,L,upper=False)       #(nb,nchannels,ndata,ndata)^{-1}(nb,nchannels,ndata,numsamples)  
        Kinv_yc = Kinv_yc.permute(0,3,2,1)                           #(nb,numsamples,ndata,nchannels)                   
        return Kinv_yc, prior_yc  

    

    
    def compute_updateterms(self,xc,yc,xt,numsamples=1,reorder=False):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
            xt: #(nb,ndata,ndim,nchannels)            
        outputs
        """        
        nb,ndata,ndim,nchannel=xc.size()
        xa_samples,prior_samples = self.sample_prior_shared(xc,xt,numsamples=numsamples)
        Kinv_yc,prior_yc = self.prepare_updateterms(xc,yc,numsamples=numsamples) 
        Kinv_yc = Kinv_yc.permute(0,2,1,3)        
        
        # xa_ : (nb,ndata2,ndim,nchannel),  xc_ : (nb,ndata,ndim,nchannel)        
        xa_ = xa_samples.unsqueeze(-1).repeat(1,1,1,nchannel)
        xc_ = xc
        #Kzx = self.kernel.eval_Kxz(xa_,xc_)     #(nb,data2,ndata,nchannels)       
        Kzx = self.kernel.eval_Kxz_ind(xa_,xc_)     #(nb,data2,ndata,nchannels)       
        
        #(nb,data2,ndata,nchannels)#(nb,data,numsamples,nchannels)=(nb,data2,numsamples,nchannels)          
        update_term = torch.einsum('bnml,bmkl->bnkl',Kzx,Kinv_yc)             
        update_term = update_term.permute(0,2,1,3)                            #(nb,numsamples,data2,nchannels)          
        density = Kzx.sum(dim=2)
        return  update_term,density,xa_samples,prior_samples,prior_yc
    
    
    def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False):
        #xa_samples,prior_samples = self.sample_prior_shared(xc,xt,numsamples=numsamples,reorder=False)  #(nb,numsamples,nchannel*(ncontext+ntarget),numchannels) 
        update_term,density,xa_samples,prior_samples,prior_yc = self.compute_updateterms(xc,yc,xt,numsamples=numsamples,reorder=False)    
        posterior_samples = prior_samples + update_term
        if reorder:            
            nchannel=posterior_samples.size(-1)    
            xa_samples,idx_reorder = xa_samples.sort(dim=1) #(nb,nchannel*(ncontext+ntarget),nchanels)            
            idx_reorder = idx_reorder[:,None,:,:].repeat(1,numsamples,1,nchannel)             
            prior_samples= torch.gather(prior_samples,2,idx_reorder)        
            posterior_samples= torch.gather(posterior_samples,2,idx_reorder)

        self.compute_regloss(yc,prior_yc)            
        return xa_samples,posterior_samples,density,prior_yc

        
    #@property
    def compute_regloss(self,yc,yc_samples):        
        """ compute yc_samples loss
        """
        reg=((yc_samples - yc[:,None,:,:])**2).sum(dim=(-2,-1)).sqrt().mean()
        self.regloss = reg
        return 

    

    
    
    

    
        
            
            
            
            
            
            



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#----------------------------
#multivariate extension
#----------------------------
class Multioutput_GPsampler(Independent_GPsampler):        
    def __init__(self,kernel, in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 20,num_sampleposterior=10, 
                      scales=.1, loglik_err=1e-2,eps=1e-6,points_per_unit=64,multiplier=2**3 ):
        super(Multioutput_GPsampler, self).__init__(kernel,in_dims,out_dims,num_channels,num_fourierbasis,num_sampleposterior,
                                                    scales,loglik_err,eps,points_per_unit,multiplier)        
        self.kernel = kernel
        self.target_idx = self.kernel.target_idx
        self.cross_idx = self.kernel.cross_idx
        
        self.w = None
        self.b = None
        self.normalizer = None
        self.inverseuse = inverseuse
    
    
        
    def sample_w_b(self,nb,nsamples,eps=1e-6):    
        """
        self.w_mu : nparams
        sample_w  : (nb,nfourifbasis,indim,nchannels)
        sample_b  : (nb,nfouribasis,indim,nchannels)        
        """        
        mu,inv_std,weight = self.kernel.prepare_cross_params() #(indims,nchannels,nchannels),(indims,nchannels,nchannels)
        eps1 = self.normal0.sample((nb,self.num_fourierbasis,self.in_dims,self.num_channels**2)).to(mu.device)
        eps2 = self.uniform0.sample((nb,self.num_fourierbasis,self.in_dims,self.num_channels**2)).to(mu.device)    
        random_w = self.normal0.sample((nb,nsamples,self.num_fourierbasis,self.num_channels**2)).to(mu.device)        
            
        
        #sample_w = mu_[None,None,:,:] + std_[None,None,:,:]*eps1    #(nb,nfouierbasis,indims,nchannels,nchannels)         
        sample_w = mu[None,None,:,:] + inv_std[None,None,:,:]*eps1    #(nb,nfouierbasis,indims,nchannels,nchannels)                 
        sample_b = eps2                                                #(nb,nfouierbasis,indims,nchannels,nchannels)
        return sample_w,sample_b,weight,random_w
    
        
        
    def sample_prior_shared(self,xc,xt,numsamples=10,reorder=False):        
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
        #xa_samples = self.samples_xa(xc,xt)                                               #(nb,nchannel*(ncontext+ntarget),ndim)  
        xa_samples =self.build_xgrid(xc,xt)
        w,b,normalizer,random_w = self.sample_w_b(nb,numsamples)                                   #(nb,nbasis,indim,numchannels**2)    
        self.w = w
        self.b = b
        self.normalizer = normalizer
        self.random_w = random_w        


        # inner product : cos in terms
        #xa_samples_ = xa_samples[:,None,:,:,None].repeat(1,1,1,1,self.num_channels)        
        xa_samples_ = xa_samples[:,None,:,:,None].repeat(1,1,1,1,self.num_channels**2)
        w = w[:,:,None,:,:]         

        #(nb,1,nchannel*(ncontext+ntarget),ndim,1)*(nb,nfourierbasis,1,ndim,numchannels)                 
        if pi2repuse:
            cos_interms = pi2*(xa_samples_*w).sum(dim=-2) + pi2*b                      
        else:
            cos_interms = (xa_samples_*w).sum(dim=-2) + pi2*b                   
            
        Psi = torch.cos(cos_interms).unsqueeze(dim=1)                                        #(nb,nfourierbasis,nchannel*(ncontext+ntarget),nunchannels)
        Psi = Psi.repeat(1,numsamples,1,1,1)                                                 #(nb,numsamples,nfourierbasis,nchannel*(ncontext+ntarget),nunchannels)
        Psi = Psi.permute(0,1,3,2,4) 
        nb,_,ndata,_,_ =Psi.size()        

        
        #scale_sigma = self.logsigma.exp().reshape(1,1,1,-1)        
        normalizer = normalizer[None,None,:,:].sqrt()                    #(1,1,1,nchannel**2)
        prior_samples = normalizer*np.sqrt(2/self.num_fourierbasis)*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w)        
        
        if reorder:
            xa_samples,idx_reorder = xa_samples.sort(dim=1)
            idx_reorder = idx_reorder[:,None,:,:] 
            prior_samples= torch.gather(prior_samples,2,idx_reorder.repeat(1,numsamples,1,nchannel))        
        return xa_samples,prior_samples     
     
        
        
        
    def sample_prior_independent(self,xc,numsamples=10):    
        """
        inputs:
            xc : (nb,ncontext,ndim,nchannel)
        outputs:
            Psi_pred : (nb,ncontext,nchannel)      # assume y-value is 1-d      
        """        
        nb = xc.size(0)        
        w = self.w[...,self.target_idx ]               #(nb,nbasis,indim,numchannels) 
        b = self.b[...,self.target_idx ]       
        normalizer = self.normalizer[...,self.target_idx]
        random_w = self.random_w[...,self.target_idx]
              
        xc_ = xc[:,None,:,:,:] #(nb,1,ndata,ndim,nchannels)
        w_ =   w[:,:,None,:,:]      
        
        if pi2repuse:
            cos_interms = pi2*(xc_*w_).sum(dim=-2) +pi2*b         
        else:
            cos_interms = (xc_*w_).sum(dim=-2) +pi2*b         
        
        
        Psi = torch.cos(cos_interms) 
        Psi = Psi.unsqueeze(dim=1)
        Psi = Psi.repeat(1,numsamples,1,1,1)                    #(nb,numsamples,nfourierbasis,nchannel*(ncontext+ntarget),nunchannels)                
        Psi = Psi.permute(0,1,3,2,4)
        _,numsample,ndata,nfbasis,nchannels = Psi.size()
        
        normalizer = normalizer[None,None,:,:].sqrt()
        prior_yc = normalizer*np.sqrt(2/self.num_fourierbasis)*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w)     #(nb,nsample,ndata,nchannel)           
        return xc,prior_yc    
                
    
    
    def prepare_updateterms(self,xc,yc,numsamples=1):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
        outputs
            L: #(nb,nchannels,ndata,ndata)
            delta_yc: #(nb,,nchannels,ndata,nsamples)            
        """
        nb,ndata,ndim,nchannel=xc.size()
        #nb,ndata,nchannel=yc.size()
        
        xc,prior_yc = self.sample_prior_independent(xc,numsamples=numsamples)
        #xc,prior_samples = self.sample_prior_independent_advanced(xc,numsamples=numsamples)
        
        Kxx = self.kernel.eval_Kxx(xc,zitter_flag=True)  #(nb,ndata,data,nchannels)              
        Kxx = Kxx.permute(0,3,1,2)              #(nb,,nchannels,ndata,data)  
        L = torch.linalg.cholesky(Kxx)                                                                             #(nb,nchannels,ndata,ndata) 
          
        likerr = (self.kernel.loglik).exp()[None,None,None,:] 
        delta_yc = yc[:,None,:,:] - (prior_yc  + likerr*torch.randn(nb,numsamples,1,nchannel).to(xc.device))         #(nb,numsample,ndata,nchannels)        
        delta_yc= delta_yc.permute(0,3,2,1)                           #(nb,nchannels,ndata,numsamples)  
        Kinv_yc = torch.cholesky_solve(delta_yc,L,upper=False)       #(nb,nchannels,ndata,ndata)^{-1}(nb,nchannels,ndata,numsamples)  
        Kinv_yc = Kinv_yc.permute(0,3,2,1)                           #(nb,numsamples,ndata,nchannels)                  
        #return Kinv_yc 
        return Kinv_yc,prior_yc 

    
    
    def compute_updateterms(self,xc,yc,xt,numsamples=1,reorder=False):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
            xt: #(nb,ndata,ndim,nchannels)            
        outputs
        """        
        nb,ndata,ndim,nchannel=xc.size()
        xa_samples,prior_samples = self.sample_prior_shared(xc,xt,numsamples=numsamples)
        Kinv_yc,prior_yc = self.prepare_updateterms(xc,yc,numsamples=numsamples)       
        
        Kinv_yc = Kinv_yc.permute(0,2,1,3)        
        Kinv_yc = Kinv_yc.repeat(1,1,1,self.num_channels)
        
        # xa_ : (nb,ndata2,ndim,nchannel**2),  xc_ : (nb,ndata,ndim,nchannel**2)
        xa_ = xa_samples.unsqueeze(-1).repeat(1,1,1,self.num_channels**2)
        xc_ = xc.repeat(1,1,1,self.num_channels)        
        
        #(nb,ndata2,ndata,nchannels**2)
        Kzx = self.kernel.eval_Kxz(xa_,xc_)                                  
        #(nb,data2,ndata,nchannels**2)x(nb,data,numsamples,nchannels**2)=(nb,data2,numsamples,nchannels**2)  
        update_term = torch.einsum('bnml,bmkl->bnkl',Kzx,Kinv_yc)             
        update_term = update_term.permute(0,2,1,3)         #(nb,numsamples,data2,nchannels**2)                            
        
        density = Kzx[...,self.target_idx].sum(dim=2)        
        return  update_term,density,xa_samples,prior_samples,prior_yc
    
  


     #reflect correlation piror on both prior and update term 
     #updated by deriviation
    def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False):      
        #nb,ndata,_,nchannels = xc.size() 
        #_,ndata2,_,_ = xt.size() 

        update_term,density,xa_samples,prior_samples,prior_yc = self.compute_updateterms(xc,yc,xt,numsamples=numsamples,reorder=False)    
        
        #nb,_,nchannel = yc.shape
        nb,_,ndata2,nchannels2 = prior_samples.shape
        nchannels = int(np.sqrt(nchannels2))

        # for example nchannel=3
        #[(0,0),(0,1),(0,2)]
        #[(1,0),(1,1),(1,2)]
        #[(2,0),(2,1),(2,2)]

        # prior term           update term
        #(0,0)            + (0,0)+(0,1)+(0,2) 
        #(1,1)            + (1,0)+(1,1)+(1,2) 
        #(2,2)            + (2,0)+(2,1)+(2,2)                

        prior_samples = prior_samples[...,self.target_idx]
        #weightratio = 1.        
        #update_term[...,self.cross_idx] = weightratio*update_term[...,self.cross_idx] 
        update_term = update_term.reshape( (nb,numsamples,ndata2,nchannels,nchannels) )
        posterior_samples = prior_samples + update_term.sum(dim=-1)          #(nb,numsamples,data2,nchannels)         
        


        
        # for example nchannel=3
        #[(0,0),(0,1),(0,2)]
        #[(1,0),(1,1),(1,2)]
        #[(2,0),(2,1),(2,2)]

        # prior term           update term
        #(0,0)            + (0,0)+(0,1)+(0,2) 
        #(1,1)            + (1,0)+(1,1)+(1,2) 
        #(2,2)            + (2,0)+(2,1)+(2,2)
                
        #weightratio = 0.05
        #prior_samples = prior_samples[...,self.target_idx]        
        #prior_samples = prior_samples.reshape((nb,numsamples,ndata2,nchannels,nchannels)).mean(dim=-1)        
        #update_term = update_term[...,self.target_idx] 
        #posterior_samples = prior_samples + update_term          #(nb,numsamples,data2,nchannels) 
        
        if reorder:            
            nchannel=posterior_samples.size(-1)    
            xa_samples,idx_reorder = xa_samples.sort(dim=1) #(nb,nchannel*(ncontext+ntarget),nchanels)            
            idx_reorder = idx_reorder[:,None,:,:].repeat(1,numsamples,1,nchannel)             
            prior_samples= torch.gather(prior_samples,2,idx_reorder)        
            posterior_samples= torch.gather(posterior_samples,2,idx_reorder)

            
            
        self.compute_regloss(yc,prior_yc)                        
        return xa_samples,posterior_samples,density,prior_yc

    
    
    
    
    
    
    
    
        
    
    
    
    


    
        
            

class Multioutput_GPsampler_V2(Multioutput_GPsampler):        
    def __init__(self,kernel, in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 20,num_sampleposterior=10, 
                      scales=.1, loglik_err=1e-2,eps=1e-6,points_per_unit=64,multiplier=2**3 ):
        super(Multioutput_GPsampler_V2, self).__init__(kernel,in_dims,out_dims,num_channels,num_fourierbasis,num_sampleposterior,
                                                    scales,loglik_err,eps,points_per_unit,multiplier)        
    
    
    #def prepare_updateterms_inversefree(self,xc,yc,numsamples=1):    
    def prepare_updateterms(self,xc,yc,numsamples=1):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
        outputs
            L: #(nb,nchannels,ndata,ndata)
            delta_yc: #(nb,,nchannels,ndata,nsamples)            
        """
        nb,ndata,ndim,nchannel=xc.size()
        nb,ndata,nchannel=yc.size()
        xc,prior_yc = self.sample_prior_independent(xc,numsamples=numsamples)
        Kxx = self.kernel.eval_Kxx(xc)
        Kxx = Kxx.permute(0,3,1,2)              #(nb,,nchannels,ndata,data)  
         
        
        likerr = (self.kernel.loglik).exp()[None,None,None,:]
        #(nb,numsample,ndata,nchannels)        
        delta_yc = yc[:,None,:,:] - (prior_yc  + likerr*torch.randn(nb,numsamples,1,nchannel).to(xc.device))      
        #(nb,numsamples,ndata,nchannels)
        delta_yc= delta_yc.permute(0,1,3,2)                             
        delta_yc = delta_yc.reshape(nb,numsamples,-1) #(nb,numsamples,ndata)
        delta_yc = delta_yc.permute(0,2,1) #(nb,ndata,numsamples)
        
        phi = self.compute_phi(xc)
        Khatinv_yc = self.compute_Khat_solver(phi,delta_yc) #(nb,ndata,numsamples) 
        Khatinv_yc = Khatinv_yc.permute(0,2,1) 
        Khatinv_yc = Khatinv_yc.reshape(nb,numsamples,nchannel,-1) 
        Khatinv_yc = Khatinv_yc.permute(0,1,3,2)         
        return Khatinv_yc,prior_yc
    
        
        
    
    def compute_updateterms(self,xc,yc,xt,numsamples=1,reorder=False):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
            xt: #(nb,ndata,ndim,nchannels)            
        outputs
        """        
        nb,ndata,ndim,nchannel=xc.size()
        xa_samples,prior_samples = self.sample_prior_shared(xc,xt,numsamples=numsamples)
        
        Kinv_yc,prior_yc = self.prepare_updateterms(xc,yc,numsamples=numsamples)       
        Kinv_yc = Kinv_yc.permute(0,2,1,3)
        Kinv_yc = Kinv_yc.repeat(1,1,1,self.num_channels)

        #print('Kinv_yc.shape {}'.format(Kinv_yc.shape))                
        #print('Kinv_yc.shape {}'.format(Kinv_yc.shape))        
        #print('Kinv_yc.shape {}'.format(Kinv_yc.shape))
        
        # xa_ : (nb,ndata2,ndim,nchannel**2),  xc_ : (nb,ndata,ndim,nchannel**2)
        xa_ = xa_samples.unsqueeze(-1).repeat(1,1,1,self.num_channels**2)
        xc_ = xc.repeat(1,1,1,self.num_channels)        
        
        Kzx = self.kernel.eval_Kxz(xa_,xc_)                                   #(nb,ndata2,ndata,nchannels**2)
        
        
        #print('Kzx.shape,Kinv_yc.shape')
        #print(Kzx.shape,Kinv_yc.shape)
        
        
        
        #(nb,data2,ndata,nchannels**2)x(nb,data,numsamples,nchannels**2)=(nb,data2,numsamples,nchannels**2)  
        update_term = torch.einsum('bnml,bmkl->bnkl',Kzx,Kinv_yc)         
        #(nb,numsamples,data2,nchannels**2)   (0,0),(0,1),(0,2),....,(2,0),(2,1),(2,2)          
        update_term = update_term.permute(0,2,1,3)                           
        density = Kzx[...,self.target_idx].sum(dim=2)        
        return  update_term,density,xa_samples,prior_samples,prior_yc
        
        
        
        
        
    def compute_phi(self,xc):            
        """SC
        inputs:
            xc : (nb,ncontext,ndim,nchannel)
        outputs:
            Psi_pred : (nb,ncontext,nchannel)      # assume y-value is 1-d      
        """        
        nb = xc.size(0)
        #w,b,normalizer,random_w = self.w, self.b, self.normalizer,self.random_w
        w = self.w[...,self.target_idx ]               #(nb,nbasis,indim,numchannels) 
        b = self.b[...,self.target_idx ]       
        normalizer = self.normalizer[...,self.target_idx]
        random_w = self.random_w[...,self.target_idx]
                     
        xc_ = xc[:,None,:,:,:]       #(nb,1,ndata,ndim,nchannels)
        w_ =   w[:,:,None,:,:]        
        if pi2repuse:        
            cos_interms_xc = pi2*(xc_*w_).sum(dim=-2) +pi2*b        
        else:
            cos_interms_xc = (xc_*w_).sum(dim=-2) +pi2*b        
            
            
        #cos_interms = pi2*(xc_*w_).sum(dim=-2) + pi2*b        
        normalizer = normalizer[None,None,:,:].sqrt()*np.sqrt(2/self.num_fourierbasis)        
        phi = normalizer*torch.cos(cos_interms_xc)                    #(nb,nfourierbasis,ncontext,nchannels)
        phi = phi.permute(0,1,3,2)
        #print(psi.shape)
        phi = phi.reshape(*phi.shape[:2],-1)
        phi = phi.permute(0,2,1)  #(nb,ndatatotal,nfbasis) 
        return phi
        
        
    def compute_Khat_solver(self,phi,delta_yc,eps=1e-4):
        loglik2 = (2*self.kernel.loglik).exp().mean()
        #
        phiphiT = torch.einsum('bnm,bmk->bnk',phi.permute(0,2,1),phi) + (loglik2+eps)*torch.eye(phi.size(-1)).to(phi.device)[None,:,:]
        L = torch.linalg.cholesky(phiphiT)
        phiTy = torch.einsum('bnm,bmk->bnk',phi.permute(0,2,1),delta_yc)
        phiphiTinv_phiTy = torch.cholesky_solve(phiTy,L,upper=False)
        phiT_phiphiTinv_phiTy = torch.einsum('bnm,bmk->bnk',phi,phiphiTinv_phiTy)

        Khatinv_yc = (1/loglik2)*(delta_yc - phiT_phiphiTinv_phiTy)
        
        #print('Khatinv_yc.shape')
        #print(Khatinv_yc.shape)        
        return  Khatinv_yc         
    
    
            
            
     #reflect correlation piror on both prior and update term 
     #updated by deriviation
    def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False):      

        update_term,density,xa_samples,prior_samples,prior_yc = self.compute_updateterms(xc,yc,xt,numsamples=numsamples,reorder=False)    
        
        #nb,_,nchannel = yc.shape
        nb,_,ndata2,nchannels2 = prior_samples.shape
        nchannels = int(np.sqrt(nchannels2))

        # for example nchannel=3
        #[(0,0),(0,1),(0,2)]
        #[(1,0),(1,1),(1,2)]
        #[(2,0),(2,1),(2,2)]

        # prior term           update term
        #(0,0)            + (0,0)+(0,1)+(0,2) 
        #(1,1)            + (1,0)+(1,1)+(1,2) 
        #(2,2)            + (2,0)+(2,1)+(2,2)                

        prior_samples = prior_samples[...,self.target_idx]
        #weightratio = 1.        
        #update_term[...,self.cross_idx] = weightratio*update_term[...,self.cross_idx] 
        update_term = update_term.reshape( (nb,numsamples,ndata2,nchannels,nchannels) )
        posterior_samples = prior_samples + update_term.sum(dim=-1)          #(nb,numsamples,data2,nchannels)         
        


        
        # for example nchannel=3
        #[(0,0),(0,1),(0,2)]
        #[(1,0),(1,1),(1,2)]
        #[(2,0),(2,1),(2,2)]

        # prior term           update term
        #(0,0)            + (0,0)+(0,1)+(0,2) 
        #(1,1)            + (1,0)+(1,1)+(1,2) 
        #(2,2)            + (2,0)+(2,1)+(2,2)
                
        #weightratio = 0.05
        #prior_samples = prior_samples[...,self.target_idx]        
        #prior_samples = prior_samples.reshape((nb,numsamples,ndata2,nchannels,nchannels)).mean(dim=-1)        
        #update_term = update_term[...,self.target_idx] 
        #posterior_samples = prior_samples + update_term          #(nb,numsamples,data2,nchannels) 
        
        if reorder:            
            nchannel=posterior_samples.size(-1)    
            xa_samples,idx_reorder = xa_samples.sort(dim=1) #(nb,nchannel*(ncontext+ntarget),nchanels)            
            idx_reorder = idx_reorder[:,None,:,:].repeat(1,numsamples,1,nchannel)             
            prior_samples= torch.gather(prior_samples,2,idx_reorder)        
            posterior_samples= torch.gather(posterior_samples,2,idx_reorder)

            
            
        self.compute_regloss(yc,prior_yc)                        
        return xa_samples,posterior_samples,density,prior_yc
            
    
    
    
            

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#--------------------------
# proxi
#    1) inverse free
#--------------------------
            
class Independent_GPsampler_Proxi(Independent_GPsampler):        
    #def __init__(self,nbasis=10,in_channels=3, nparams = 3, w_mu = None,w_std= None):
    def __init__(self,kernel=None,in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 20,num_sampleposterior=10 ,
                      scales=.1, loglik_err=1e-2, eps=1e-6,points_per_unit=64,multiplier=2**3):
        super(Independent_GPsampler_Proxi, self).__init__(kernel,in_dims,out_dims,num_channels,num_fourierbasis,num_sampleposterior,
                                                          scales, loglik_err, eps,points_per_unit,multiplier)
        #super(Multioutput_GPsampler, self).__init__()
        

#     # inverse relaxation
#     def prepare_updateterms_inversefree(self,xc,yc,numsamples=1):    
#     #def prepare_updateterms(self,xc,yc,numsamples=1):
#         """
#         inputs
#             xc: #(nb,ndata,ndim,nchannels)
#             yc: #(nb,ndata,nchannels)
#         outputs
#             L: #(nb,nchannels,ndata,ndata)
#             delta_yc: #(nb,,nchannels,ndata,nsamples)            
#         """
#         nb,ndata,ndim,nchannel=xc.size()
#         nb,ndata,nchannel=yc.size()
#         xc,prior_samples = self.sample_prior_independent(xc,numsamples=numsamples)
#         Kxx = self.kernel.eval_Kxx(xc)
#         Kxx = Kxx.permute(0,3,1,2)              #(nb,,nchannels,ndata,data)  
#         Kxx_diag = Kxx.diagonal(dim1=-2,dim2=-1) #(nb,,nchannels,ndata)  
        
#         likerr = (self.kernel.loglik).exp()[None,None,None,:] 
#         delta_yc = yc[:,None,:,:] - (prior_samples  + likerr*torch.randn(1,numsamples,1,nchannel).to(xc.device))         #(nb,numsample,ndata,nchannels)        
#         delta_yc= delta_yc.permute(0,3,2,1)                           #(nb,nchannels,ndata,numsamples)  
#         #Kinv_yc = torch.cholesky_solve(delta_yc,L,upper=False)       #(nb,nchannels,ndata,ndata)^{-1}(nb,nchannels,ndata,numsamples)

#         #print('compute inverse free indepedent')        
#         Kinv_yc = (1/(Kxx_diag[:,:,:,None]+eps))*delta_yc            #'compute inverse free multi correlated'    
#         Kinv_yc = Kinv_yc.permute(0,3,2,1)                           #(nb,numsamples,ndata,nchannels)                  
#         return Kinv_yc 


        
        
    # inverse relaxation
    def prepare_updateterms(self,xc,yc,numsamples=1):    
    #def prepare_updateterms(self,xc,yc,numsamples=1):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
        outputs
            L: #(nb,nchannels,ndata,ndata)
            delta_yc: #(nb,,nchannels,ndata,nsamples)            
        """
        nb,ndata,ndim,nchannel=xc.size()
        nb,ndata,nchannel=yc.size()
        xc,prior_yc = self.sample_prior_independent(xc,numsamples=numsamples)
        Kxx = self.kernel.eval_Kxx(xc)
        Kxx = Kxx.permute(0,3,1,2)              #(nb,,nchannels,ndata,data)  
        Kxx_diag = Kxx.diagonal(dim1=-2,dim2=-1) #(nb,,nchannels,ndata)  
        
        likerr = (self.kernel.loglik).exp()[None,None,None,:] 
        delta_yc = yc[:,None,:,:] - (prior_yc  + likerr*torch.randn(1,numsamples,1,nchannel).to(xc.device))         #(nb,numsample,ndata,nchannels)        
        delta_yc= delta_yc.permute(0,3,2,1)                           #(nb,nchannels,ndata,numsamples)  
        #Kinv_yc = torch.cholesky_solve(delta_yc,L,upper=False)       #(nb,nchannels,ndata,ndata)^{-1}(nb,nchannels,ndata,numsamples)

        #print('compute inverse free indepedent')        
        Kinv_yc = (1/(Kxx_diag[:,:,:,None]+eps))*delta_yc            #'compute inverse free multi correlated'    
        Kinv_yc = Kinv_yc.permute(0,3,2,1)                           #(nb,numsamples,ndata,nchannels)                  
        return Kinv_yc ,prior_yc

    
        

        
        
        
        
class Multioutput_GPsampler_Proxi(Multioutput_GPsampler):        
    def __init__(self,kernel, in_dims=1,out_dims=1,num_channels=3, num_fourierbasis = 20,num_sampleposterior=10, 
                      scales=.1, loglik_err=1e-2,eps=1e-6,points_per_unit=64,multiplier=2**3 ):
        super(Multioutput_GPsampler_Proxi, self).__init__(kernel,in_dims,out_dims,num_channels,num_fourierbasis,num_sampleposterior,
                                                    scales,loglik_err,eps,points_per_unit,multiplier)        
    
    
    #inverse free
    #def prepare_updateterms_inversefree(self,xc,yc,numsamples=1):    
    def prepare_updateterms(self,xc,yc,numsamples=1):
        """
        inputs
            xc: #(nb,ndata,ndim,nchannels)
            yc: #(nb,ndata,nchannels)
        outputs
            L: #(nb,nchannels,ndata,ndata)
            delta_yc: #(nb,,nchannels,ndata,nsamples)            
        """
        nb,ndata,ndim,nchannel=xc.size()
        nb,ndata,nchannel=yc.size()
        xc,prior_yc = self.sample_prior_independent(xc,numsamples=numsamples)
        Kxx = self.kernel.eval_Kxx(xc)
        Kxx = Kxx.permute(0,3,1,2)              #(nb,,nchannels,ndata,data)  
        Kxx_diag = Kxx.diagonal(dim1=-2,dim2=-1) #(nb,,nchannels,ndata)  
        
        likerr = (self.kernel.loglik).exp()[None,None,None,:] 
        delta_yc = yc[:,None,:,:] - (prior_yc  + likerr*torch.randn(nb,numsamples,1,nchannel).to(xc.device))         #(nb,numsample,ndata,nchannels)        
        delta_yc= delta_yc.permute(0,3,2,1)                           #(nb,nchannels,ndata,numsamples)  
        
        #Kinv_yc = torch.cholesky_solve(delta_yc,L,upper=False)       #(nb,nchannels,ndata,ndata)^{-1}(nb,nchannels,ndata,numsamples)
        Kinv_yc = (1/(Kxx_diag[:,:,:,None]+eps))*delta_yc            #'compute inverse free multi correlated' 
        Kinv_yc = Kinv_yc.permute(0,3,2,1)                           #(nb,numsamples,ndata,nchannels)                  
        return Kinv_yc,prior_yc 
        
        
        
        
        
        
        
        
        
        

        
#     #reflect correlation piror on only prior and indepdenet update term 
#     #def sample_posterior_proxi(self,xc,yc,xt,numsamples=1,reorder=False):           
#     def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False):        
#         update_term,density,xa_samples,prior_samples = self.compute_updateterms(xc,yc,xt,numsamples=numsamples,reorder=False)    

#         nb,_,ndata,_ = prior_samples.size()         
#         prior_samples = prior_samples.reshape(nb,numsamples,ndata,self.num_channels,self.num_channels)
#         #update_term = update_term.reshape(nb,numsamples,ndata,self.num_channels,self.num_channels)
        
#         # for example nchannel=3
#         #[(0,0),(0,1),(0,2)]
#         #[(1,0),(1,1),(1,2)]
#         #[(2,0),(2,1),(2,2)]

#         # prior term              update term
#         #(0,0)+(0,1)+(0,2)      + (0,0)
#         #(1,0)+(1,1)+(1,2)      + (1,1)
#         #(2,0)+(2,1)+(2,2)      + (2,2)
        
#         #posterior_samples = prior_samples.sum(dim=-1) + update_term[...,self.target_idx]          #(nb,numsamples,data2,nchannels**2) 
#         posterior_samples = prior_samples.mean(dim=-1) + update_term[...,self.target_idx]          #(nb,numsamples,data2,nchannels**2) 
        
#         if reorder:            
#             nchannel=posterior_samples.size(-1)    
#             xa_samples,idx_reorder = xa_samples.sort(dim=1) #(nb,nchannel*(ncontext+ntarget),nchanels)            
#             idx_reorder = idx_reorder[:,None,:,:].repeat(1,numsamples,1,nchannel)             
#             prior_samples= torch.gather(prior_samples,2,idx_reorder)        
#             posterior_samples= torch.gather(posterior_samples,2,idx_reorder)

#         return xa_samples,posterior_samples,density        
        
        
        
#     def sample_prior_independent(self,xc,numsamples=10):    
#         """
#         inputs:
#             xc : (nb,ncontext,ndim,nchannel)
#         outputs:
#             Psi_pred : (nb,ncontext,nchannel)      # assume y-value is 1-d      
#         """        
#         #nb = xc.size(0)        
#         nb,ndata,ndim,nchannel=xc.size()

#         #w = self.w[...,self.target_idx ]               #(nb,nbasis,indim,numchannels) 
#         #b = self.b[...,self.target_idx ]       
#         #normalizer = self.normalizer[...,self.target_idx]
#         #random_w = self.random_w[...,self.target_idx]
#         w = self.w 
#         b = self.b
#         normalizer = self.normalizer
#         random_w = self.random_w 
            
            
#         #xc_ = xc[:,None,:,:,:] #(nb,1,ndata,ndim,nchannels)
#         xc_ = xc[:,None,:,:,:].repeat(1,1,1,1,nchannel)  #(nb,1,ndata,ndim,nchannels**2)
#         w_ =   w[:,:,None,:,:]                
#         cos_interms = (xc_*w_).sum(dim=-2) +pi2*b         
#         print('cos_interms.shape in advanced')
#         print(cos_interms.shape)

        
        
#         Psi = torch.cos(cos_interms) 
#         Psi = Psi.unsqueeze(dim=1)
#         Psi = Psi.repeat(1,numsamples,1,1,1)                                                 #(nb,numsamples,nfourierbasis,nchannel*(ncontext+ntarget),nunchannels)                
#         Psi = Psi.permute(0,1,3,2,4)
#         _,_,ndata,nfbasis,nchannels = Psi.size()
        
#         normalizer = normalizer[None,None,:,:].sqrt()
#         prior_samples = normalizer*np.sqrt(2/self.num_fourierbasis)*torch.einsum('bsnmc,bsmc->bsnc',Psi,random_w)     #(nb,nsample,ndata,nchannel)        
#         print('prior_samples in advanced')
#         print(prior_samples.shape)
        
        
#         #return xc,prior_samples   
#         prior_samples = prior_samples.reshape(nb,numsamples,ndata,nchannel,nchannel)
#         #prior_samples = prior_samples.sum(dim=-1)
#         prior_samples = prior_samples.mean(dim=-1)        
#         return xc,prior_samples    
        
    
    

    
            
        