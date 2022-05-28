from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn as nn
import torch.nn.functional as F  
import math
import numpy as np
import torch
import itertools as it
from attrdict import AttrDict

from convcnp.utils import init_sequential_weights, to_multiple        
from test_kernels import SM_kernel        
from test_gpsampler import Independent_GPsampler     
from test_gpsampler import Multioutput_GPsampler,Multioutput_GPsampler2,Multioutput_GPsampler_Proxi,Multioutput_GPsampler_Cat        
#from test_gpsampler2  import Spikeslab_GPsampler
#from test_gpsampler3  import NeuralSpikeslab_GPsampler

#from test_gpsampler3  import Spikeslab_GPsampler  # multi channel results 
#from test_gpsampler5  import Spikeslab_GPsampler
from test_gpsampler7  import Spikeslab_GPsampler


from test_cnnmodels import get_cnnmodels
#from test_ind_correlatenp import ConvDeepset
from test_ind_correlatenp import eps,num_basis,num_fourierbasis,loglik_err
from test_ind_correlatenp import ICGP_Convnp,ConvDeepset,compute_loss_gp
    
    

class DCGP_Convnp(ICGP_Convnp):
    def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
                      cnntype='shallow',num_postsamples=10,init_lengthscale=1.0):
        super(DCGP_Convnp,self).__init__(in_dims,out_dims, num_channels,
                                          cnntype,num_postsamples,init_lengthscale)
        
        self.modelname = 'gpdep'        
#         self.in_dims = in_dims
#         self.out_dims = out_dims       
#         self.num_channels = num_channels
#         self.num_samples = num_postsamples
#         self.cnntype = cnntype
#         self.cnn = get_cnnmodels(cnntype)
                
#         kernel = SM_kernel(in_dims=in_dims,num_channels=num_channels,scales=init_lengthscale,loglik_err=loglik_err)
# #         self.gpsampler = Multioutput_GPsampler(kernel, 
# #                                                in_dims=in_dims,
# #                                                out_dims=out_dims,
# #                                                num_channels=num_channels, 
# #                                                num_fourierbasis = num_fourierbasis,
# #                                                points_per_unit=self.cnn.points_per_unit,
# #                                                multiplier=self.cnn.multiplier)
#         print('more randomness')
#         self.gpsampler = Multioutput_GPsampler2(kernel, 
#                                                in_dims=in_dims,
#                                                out_dims=out_dims,
#                                                num_channels=num_channels, 
#                                                num_fourierbasis = num_fourierbasis,
#                                                points_per_unit=self.cnn.points_per_unit,
#                                                multiplier=self.cnn.multiplier)
        
#         print('spikeslab prior ')    
#         self.gpsampler =  Spikeslab_GPsampler(in_dims=in_dims,
#                                             out_dims=out_dims,
#                                             num_channels=num_channels, 
#                                             num_fourierbasis = num_fourierbasis,
#                                             scales=init_lengthscale,
#                                             loglik_err=loglik_err,
#                                             points_per_unit=self.cnn.points_per_unit,
#                                             multiplier=self.cnn.multiplier)

#         print('Neuralspikeslab prior ')    
#         self.gpsampler =  NeuralSpikeslab_GPsampler(in_dims=in_dims,
#                                             out_dims=out_dims,
#                                             num_channels=num_channels, 
#                                             num_fourierbasis = num_fourierbasis,
#                                             scales=init_lengthscale,
#                                             loglik_err=loglik_err,
#                                             points_per_unit=self.cnn.points_per_unit,
#                                             multiplier=self.cnn.multiplier)









#         print('spikeslab prior fixed ')    
#         self.gpsampler =  Spikeslab_GPsampler(in_dims=in_dims,
#                                             out_dims=out_dims,
#                                             num_channels=num_channels, 
#                                             num_fourierbasis = num_fourierbasis,
#                                             scales=init_lengthscale,
#                                             loglik_err=loglik_err,
#                                             points_per_unit=self.cnn.points_per_unit,
#                                             multiplier=self.cnn.multiplier,
#                                             useweightnet = False)

        
        
        #print('Neuralspikeslab prior ')    
        self.gpsampler =  Spikeslab_GPsampler(in_dims=in_dims,
                                            out_dims=out_dims,
                                            num_channels=num_channels, 
                                            num_fourierbasis = num_fourierbasis,
                                            scales=init_lengthscale,
                                            loglik_err=loglik_err,
                                            points_per_unit=self.cnn.points_per_unit,
                                            multiplier=self.cnn.multiplier,
                                            useweightnet = True)




        
#         gp_linear = nn.Sequential(nn.Linear(2*self.num_channels,8))
#         self.gp_linear = init_sequential_weights(gp_linear)
#         self.num_basis = num_basis
#         self.num_features = num_channels*num_basis
#         cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_features))              
#         self.cnn_linear = init_sequential_weights(cnn_linear)
        
        
#         #self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels)
#         self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels,length_scales = init_lengthscale)
        
        
#         #pred
#         self.pred_linear = init_sequential_weights(nn.Sequential(nn.Linear(self.num_features,2*self.num_channels)))
        

        
        
        
        
        
        
class DCGPCAT_Convnp(ICGP_Convnp):
    def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
                      cnntype='shallow',num_postsamples=10,init_lengthscale=0.1):
        super(DCGPCAT_Convnp,self).__init__(in_dims,out_dims, num_channels,
                                          cnntype,num_postsamples,init_lengthscale)
        
        self.modelname = 'gpdep2'        
#         self.in_dims = in_dims
#         self.out_dims = out_dims       
#         self.num_channels = num_channels
#         self.num_samples = num_postsamples
#         self.cnntype = cnntype
#         self.cnn = get_cnnmodels(cnntype)
                
#         #print('DCGPCAT_Convnp,init_lengthscale,loglik_er')
#         #print(init_lengthscale,loglik_err)
        kernel = SM_kernel(in_dims=in_dims,num_channels=num_channels,scales=init_lengthscale,loglik_err=loglik_err)
        self.gpsampler = Multioutput_GPsampler_Cat(kernel, 
                                                   in_dims=in_dims,
                                                   out_dims=out_dims,
                                                   num_channels=num_channels, 
                                                   num_fourierbasis = num_fourierbasis,
                                                   points_per_unit=self.cnn.points_per_unit,
                                                   multiplier=self.cnn.multiplier)
        
        #gp_linear = nn.Sequential(nn.Linear(2*self.num_channels,8))
        gp_linear = nn.Sequential(nn.Linear(self.num_channels**2 + self.num_channels,8))        
        self.gp_linear = init_sequential_weights(gp_linear)

        self.num_basis = num_basis
        self.num_features = num_channels*num_basis
        cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_features))              
        self.cnn_linear = init_sequential_weights(cnn_linear)
        
        
        #self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels)
        self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels,length_scales = init_lengthscale)
        
        #pred
        self.pred_linear = init_sequential_weights(nn.Sequential(nn.Linear(self.num_features,2*self.num_channels)))
        
        
        

        
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
# class DCGPCAT_Convnp(ICGP_Convnp):
#     def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
#                       cnntype='shallow',num_postsamples=10,init_lengthscale=0.1):
#         super(DCGPCAT_Convnp,self).__init__(in_dims,out_dims, num_channels,
#                                           cnntype,num_postsamples,init_lengthscale)
        
#         self.modelname = 'gpdep2'        
#         self.in_dims = in_dims
#         self.out_dims = out_dims       
#         self.num_channels = num_channels
#         self.num_samples = num_postsamples
#         self.cnntype = cnntype
#         self.cnn = get_cnnmodels(cnntype)
                
#         #print('DCGPCAT_Convnp,init_lengthscale,loglik_er')
#         #print(init_lengthscale,loglik_err)
#         kernel = SM_kernel(in_dims=in_dims,num_channels=num_channels,scales=init_lengthscale,loglik_err=loglik_err)
#         self.gpsampler = Multioutput_GPsampler_Cat(kernel, 
#                                                    in_dims=in_dims,
#                                                    out_dims=out_dims,
#                                                    num_channels=num_channels, 
#                                                    num_fourierbasis = num_fourierbasis,
#                                                    points_per_unit=self.cnn.points_per_unit,
#                                                    multiplier=self.cnn.multiplier)
        
#         #gp_linear = nn.Sequential(nn.Linear(2*self.num_channels,8))
#         gp_linear = nn.Sequential(nn.Linear(self.num_channels**2 + self.num_channels,8))        
#         self.gp_linear = init_sequential_weights(gp_linear)

#         self.num_basis = num_basis
#         self.num_features = num_channels*num_basis
#         cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_features))              
#         self.cnn_linear = init_sequential_weights(cnn_linear)
        
        
#         #self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels)
#         self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels,length_scales = init_lengthscale)

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


#     def sample_functionalfeature(self,xc,yc,xt,numsamples=1):
#         xa,post_samples,density,yc_samples = self.gpsampler.sample_posterior(xc,yc,xt,numsamples=numsamples,reorder=True)
#         return post_samples
    
                    
        
        
#------------------------------------------------------------------------------------------        
        
        
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


#     def sample_functionalfeature(self,xc,yc,xt,numsamples=1):
#         xa,post_samples,density,yc_samples = self.gpsampler.sample_posterior(xc,yc,xt,numsamples=numsamples,reorder=True)
#         return post_samples
    
            



# def compute_loss_gp( pred_mu,pred_std, target_y, z_samples=None, qz_c=None, qz_ct=None):
    
#     """
#     compute NLLLossLNPF
#     # computes approximate LL in a numerically stable way
#     # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
#     # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
#     # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
#     # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
#     # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)
    
#     """
    
#     def sum_from_nth_dim(t, dim):
#         """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
#         return t.view(*t.shape[:dim], -1).sum(-1)


#     def sum_log_prob(prob, sample):
#         """Compute log probability then sum all but the z_samples and batch."""    
#         log_p = prob.log_prob(sample)          # size = [n_z_samples, batch_size, *]    
#         sum_log_p = sum_from_nth_dim(log_p, 2) # size = [n_z_samples, batch_size]
#         return sum_log_p

    
#     p_yCc = Normal(loc=pred_mu, scale=pred_std)    
#     if qz_c is not None:
#         qz_c = Normal(loc=qz_c[0], scale=qz_c[1])
        
#     if qz_ct is not None:
#         qz_ct = Normal(loc=qz_ct[0], scale=qz_ct[1])
        
        
#     n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape    
#     # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
#     sum_log_p_yCz = sum_log_prob(p_yCc, target_y)

    
#     # uses importance sampling weights if necessary
#     if z_samples is not None:
#         # All latents are treated as independent. size = [n_z_samples, batch_size]
#         sum_log_qz_c = sum_log_prob(qz_c, z_samples)
#         sum_log_qz_ct = sum_log_prob(qz_ct, z_samples)
#         # importance sampling : multiply \prod_t p(y^t|z)) by q(z|y_cntxt) / q(z|y_cntxt, y_trgt)
#         # i.e. add log q(z|y_cntxt) - log q(z|y_cntxt, y_trgt)
#         #print(sum_log_p_yCz, sum_log_qz_c, sum_log_qz_ct)
#         sum_log_w_k = sum_log_p_yCz + sum_log_qz_c - sum_log_qz_ct
#     else:
#         sum_log_w_k = sum_log_p_yCz

#     # log_sum_exp_z ... . size = [batch_size]
#     log_S_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)
#     # - log(n_z_samples)
#     log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)    

#     #print('log_E_z_sum_p_yCz {}'.format(log_E_z_sum_p_yCz.mean().item()))
#     # NEGATIVE log likelihood
#     #return -log_E_z_sum_p_yCz
#     return -log_E_z_sum_p_yCz.mean()  #averages each loss over batches 



