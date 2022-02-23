from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn as nn
import torch.nn.functional as F  
import math
import numpy as np
import torch
import itertools as it

from convcnp.utils import init_sequential_weights, to_multiple        

from test_kernels import SM_kernel        
from test_gpsampler import Independent_GPsampler,Multioutput_GPsampler      
from test_gpsampler import Independent_GPsampler_Proxi,Multioutput_GPsampler_Proxi        

from test_cnnmodels import get_cnnmodels
from test_ind_correlatenp import eps,num_basis,num_fourierbasis,loglik_err
from test_ind_correlatenp import compute_loss_gp
from test_ind_correlatenp import ICGP_Convnp,ConvDeepset
from test_ind_correlatenp import collapse_z_samples,replicate_z_samples


# eps=1e-6    
# num_base = 5
# num_fourierbase = 20
# loglik_err = 0.1




# def collapse_z_samples(t):
#     """Merge n_z_samples and batch_size in a single dimension."""
#     n_z_samples, batch_size, *rest = t.shape
#     return t.contiguous().view(n_z_samples * batch_size, *rest)


# def replicate_z_samples(t, n_z_samples):
#     """Replicates a tensor `n_z_samples` times on a new first dim."""
#     nb,*nleft = t.shape
#     t.unsqueeze_(dim=1)
#     return t.expand(nb, n_z_samples, *nleft)
#     #return t.repeat(1, n_z_samples, *nleft)

        
exact_sampler = True    
class DCGP_Convnp(ICGP_Convnp):
    def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
                      cnntype='shallow',num_postsamples=10,init_lengthscale=0.1):
        super(DCGP_Convnp,self).__init__(in_dims,out_dims, num_channels,
                                          cnntype,num_postsamples,init_lengthscale)
        
        self.modelname = 'gpdep'        
        kernel = SM_kernel(in_dims=in_dims,num_channels=num_channels,scales=init_lengthscale,loglik_err=loglik_err)
        #self.gpsampler = Multioutput_GPsampler(kernel, in_dims=in_dims,out_dims=out_dims,num_channels=num_channels, 
        #                                       num_fourierbasis = num_fourierbasis,
        #                                       points_per_unit=self.cnn.points_per_unit,
        #                                        multiplier=self.cnn.multiplier)
        
        if exact_sampler==True:
            self.samplertype='exact'
            self.gpsampler = Multioutput_GPsampler(kernel, 
                                                   in_dims=in_dims,
                                                   out_dims=out_dims,
                                                   num_channels=num_channels,
                                                   num_fourierbasis = num_fourierbasis,
                                                   points_per_unit=self.cnn.points_per_unit,
                                                   multiplier=self.cnn.multiplier)

        else:
            self.samplertype='proxi'
            self.gpsampler =  Multioutput_GPsampler_Proxi( kernel, 
                                                           in_dims=in_dims,
                                                           out_dims=out_dims,
                                                           num_channels=num_channels,
                                                           num_fourierbasis = num_fourierbasis,
                                                           points_per_unit=self.cnn.points_per_unit,
                                                           multiplier=self.cnn.multiplier)

        #print(self.gpsampler)
        
# class DCGP_Convnp(nn.Module):
#     def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
#                       cnntype='shallow',num_postsamples=10,init_lengthscale=0.1):
#         super(DCGP_Convnp,self).__init__()
        
#         self.modelname = 'gpdep'        
#         self.in_dims = in_dims
#         self.out_dims = out_dims       
#         self.num_channels = num_channels
#         self.num_samples = num_postsamples

        
#         self.cnntype = cnntype
#         self.cnn = get_cnnmodels(cnntype)
                

#         kernel = SM_kernel(in_dims=in_dims,num_channels=num_channels,scales=init_lengthscale,loglik_err=loglik_err)
#         self.gpsampler = Multioutput_GPsampler(kernel, in_dims=in_dims,out_dims=out_dims,num_channels=num_channels, 
#                                                num_fourierbasis = num_fourierbasis,
#                                                points_per_unit=self.cnn.points_per_unit,
#                                                multiplier=self.cnn.multiplier)
        
#         gp_linear = nn.Sequential(nn.Linear(2*self.num_channels,8))
#         self.gp_linear = init_sequential_weights(gp_linear)

#         self.num_basis = num_basis
#         self.num_features = num_channels*num_basis
#         cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_features))              
#         self.cnn_linear = init_sequential_weights(cnn_linear)
        
        
#         self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels)
#         pred_linear = nn.Sequential(nn.Linear(self.num_features,2*self.num_channels))
#         self.pred_linear = init_sequential_weights(pred_linear)
        

        
        
#     def forward(self,xc,yc,xt,yt=None):        

#         nb,ndata,ndim,nchannel = xc.size()
#         _ ,ndata2,_,_ = xt.size()
                
#         xa_samples,post_samples,density,yc_samples = self.gpsampler.sample_posterior(xc,yc,xt,reorder=False,numsamples=self.num_samples)       
#         density_samples = replicate_z_samples(density,self.num_samples)       
#         features = torch.cat([post_samples,density_samples],dim=-1)
#         features = self.gp_linear(features)
#         _,_,ndata,nchannel = features.size()
#         features = features.reshape(-1,ndata,nchannel)
        
        
        
        
#         features_update = self.cnn(features.permute(0,2,1))
#         features_update = self.cnn_linear(features_update.permute(0,2,1))
#         features_update = features_update.reshape(nb,self.num_samples,ndata,self.num_basis,self.num_channels)
                
#         xt = replicate_z_samples(xt,self.num_samples)        
#         xa_samples = replicate_z_samples(xa_samples,self.num_samples)
#         xa_samples = xa_samples.unsqueeze(-1).repeat(1,1,1,1,self.num_channels)
        
#         #print('features_update.size(),xa_samples.size(),xt.size()')        
#         #print(features_update.size(),xa_samples.size(),xt.size())
        
#         xt = collapse_z_samples(xt)
#         xa_samples = collapse_z_samples(xa_samples)       
#         features_update = collapse_z_samples(features_update)
        
#         #print('xa_samples.shape,features_update.shape,xt.shape')
#         #print( xa_samples.shape,features_update.shape,xt.shape)
        
#         #smooth feature
#         smoothed_features_update = self.smoother(xa_samples,features_update,xt )              
#         smoothed_features_update = smoothed_features_update.reshape(nb,self.num_samples,ndata2,-1)
#         smoothed_features_update = smoothed_features_update.permute(1,0,2,3)
        
#         #predict        
#         features_out = self.pred_linear(smoothed_features_update)                
#         pmu,plogstd = features_out.split((self.num_channels,self.num_channels),dim=-1)            
        
#         return pmu, 0.01+0.99*F.softplus(plogstd)
    
    
#     @property
#     def num_params(self):
#         """Number of parameters in model."""
#         return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])


#     def compute_regloss_terms(self):
#         regtotal = self.gpsampler.regloss
#         return regtotal

    
            

        