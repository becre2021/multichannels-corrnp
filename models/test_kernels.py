# from torch.autograd import Variable
# from torch.distributions.normal import Normal
# from torch.distributions.uniform import Uniform

# import torch
# import torch.nn as nn

# import numpy as np

# import math
# import itertools as it

# pi2 = 2*math.pi
# #eps=1e-6
# eps=1e-16




# pi2repuse = True
# class Multioutput_kernel(nn.Module):
#     def __init__(self,in_dims=1,num_channels=3):
#         super(Multioutput_kernel,self).__init__()
#         self.in_dims = in_dims
#         self.num_channels = num_channels        
        
#         self.product_list = list(it.product(list(np.arange(self.num_channels)),repeat = 2))
#         self.target_idx = [idx  for idx,(ii,jj) in enumerate(self.product_list) if ii==jj]
#         self.cross_idx = np.setdiff1d(np.arange(num_channels**2),self.target_idx).tolist()
        
#         self.logmu_prior =    nn.Parameter( torch.log(eps  + 0.0*torch.ones(in_dims,num_channels))  ,requires_grad = False)                       
#         #self.logstd_prior =   nn.Parameter( torch.log(eps + 0.05*torch.ones(in_dims,num_channels)) ,requires_grad = False )                  
#         self.logstd_prior =   nn.Parameter( torch.log(eps + 0.1*torch.ones(in_dims,num_channels)) ,requires_grad = False )          
#         return
                        
        
        
        
#     def eval_Kxz(x,z=None):
#         raise NotImplementedError
        
    
#     def eval_Kxz_prior(self,x1,x2=None,eps=1e-6):
#         if x2 is None:
#             x2 = x1    
#         else:
#             assert(x1.dim()==x2.dim())
#         # Compute shapes.            
#         #nbatch,npoints,nchannel = x1.size()
#         nb,ndata2,ndim,nchannel = x1.size()        
#         inv_std = 1/(self.logstd_prior.exp()+eps)
        
#         exp_x1 = x1*inv_std
#         exp_x2 = x2*inv_std
        
#         exp_x1_2 = torch.pow(exp_x1[:,:,None,:],2).sum(dim=-2) 
#         exp_x2_2 = torch.pow(exp_x2[:,None,:,:],2).sum(dim=-2)         
#         exp_term = exp_x1_2 + exp_x2_2  -2*torch.einsum('bijc,bjkc->bikc',exp_x1,exp_x2.permute(0,2,1,3))        
        
#         Kxz = torch.exp(-0.5*exp_term**2)   
#         return Kxz
            


        
        
# class SM_kernel(Multioutput_kernel):
#     def __init__(self,in_dims=1,num_channels=3, scales=0.1,loglik_err=0.1): #basic setting
#     #def __init__(self,in_dims=1,num_channels=3, scales=0.1,loglik_err=1.):
        
#         super(SM_kernel,self).__init__(in_dims,num_channels)
                        
#         self.logsigma = nn.Parameter( torch.log(eps + torch.ones(num_channels)) ,requires_grad=False)         
        
#         #self.logmu =    nn.Parameter( torch.log(eps  + eps*torch.rand(in_dims,num_channels)) )     
#         #self.logmu =    nn.Parameter( torch.log(eps  +  .1*torch.rand(in_dims,num_channels)) )  #much powerful                         
#         self.logmu =    nn.Parameter( torch.log(eps  +  1.*torch.rand(in_dims,num_channels)) )  #much powerful                                        
        
#         self.logstd =   nn.Parameter( torch.log(eps + scales*torch.ones(in_dims,num_channels)) )  
#         #self.logweight =   nn.Parameter( torch.log(eps + 1*torch.ones(num_channels)) )
        
#         self.loglik =   nn.Parameter( torch.log(eps + loglik_err*torch.ones(num_channels)) )
#         self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
#         #self.product_list = list(it.product(list(np.arange(self.num_channels)),repeat = 2))
#         #self.target_idx = [idx  for idx,(ii,jj) in enumerate(self.product_list) if ii==jj]
#         #self.cross_idx = np.setdiff1d(np.arange(num_channels**2),i_gpsampler.kernel.target_idx).tolist()
    

#     #v4
#     def prepare_cross_params(self,eps=1e-6):
#         num_channels = self.num_channels
#         mu = self.logmu.exp()
#         std = self.logstd.exp()
#         weight = self.logsigma.exp() #sqrt
        
#         ndim = self.in_dims

#         cross_mu,cross_std,cross_weight = [],[],[]
#         #product_list = list(it.product(list(np.arange(num_channels)),repeat = 2))

#         for (j,i) in self.product_list:
#             if j!=i:            
#                 # std cross
#                 std_j_inv = 1/(std[:,j]+eps)
#                 std_i_inv = 1/(std[:,i]+eps)
#                 std_ji_inv = std_j_inv+std_i_inv
#                 std_ji = 1/(std_ji_inv+eps)

#                 # mu cross            
#                 mu_j = mu[:,j]
#                 mu_i = mu[:,i]
#                 mu_ji = std_ji*(std_j_inv*mu_j + std_i_inv*mu_i)

#                 # weight
#                 std_ji = std[:,j] + std[:,i] 
#                 exp_in = ((mu_j-mu_i)/std_ji)**2            
#                 determinant = (pi2**self.in_dims)*(std_ji**2).prod() + eps
#                 #determinant = (pi2**self.in_dims)*((std_ji + 1.)**2).prod() #original      
#                 #determinant = (pi2**self.in_dims)*((std_ji + 0.5)**2).prod()      
                
#                 #determinant = (pi2**self.in_dims)*((std_ji + 0.5)**2).prod()                
#                 #determinant =  torch.tensor(1.0).to(mu.device)
#                 determinant = torch.sqrt(determinant)
                
#                 #weight_ji = (1/determinant)*torch.exp(-0.5*exp_in.sum())     
#                 #weight_ji = torch.exp(-0.5*exp_in.sum())                     
#                 #weight_ji = torch.tensor(1.0).to(mu.device) #v44 not works
#                 weight_ji = (1/determinant)*torch.exp(-0.5*exp_in.sum())     
                
#                 #weight_ji = (1/determinant)*torch.exp(-0.5*exp_in.sum())*torch.sqrt(weight[j]*weight[i])                  
#                 #print('learnable weight j!=i, ji {}'.format(weight_ji))
#             else:
#                 mu_ji = mu[:,j]
#                 std_ji = std[:,j]
#                 weight_ji = torch.tensor(1.0).to(mu.device)
#                 #weight_ji = torch.sqrt(weight[j])
#                 #print('learnable weight j==i, ji {}'.format(weight_ji))
                
#             cross_mu.append(mu_ji)
#             cross_std.append(std_ji)
#             cross_weight.append(weight_ji[None])

#         cross_mu = torch.cat(cross_mu).reshape(ndim,num_channels**2)
#         cross_std = torch.cat(cross_std).reshape(ndim,num_channels**2)
#         cross_weight = torch.cat(cross_weight).reshape(1,num_channels**2)
                
#         cross_weight = torch.clamp(cross_weight,min=eps,max=1.0)            
#         return cross_mu,1/(cross_std+eps),cross_weight



            

#     #def eval_Kxx(self,xc,zitter=1e-6,zitter_flag = True):
#     def eval_Kxx(self,xc,zitter=1e-4,zitter_flag = True):
        
#         """
#         inputs:
#             xc : (nb,ncontext,ndim,nchannel)
#         outputs:
#             Kxz : (nb,ncontext,ncontext,nchannel)      # assume y-value is 1-d      
#         """
#         #print(len(xc.shape))
#         assert len(xc.shape) == 4 and xc.size(-1) == self.num_channels                        
        
#         #check dataset        
#         nb,ndata,ndim,nchannels = xc.size()                            
#         xt = xc
        
#         #get model parameters
#         mu,inv_std,weight=self.prepare_cross_params()
#         mu = mu[...,self.target_idx]
#         inv_std = inv_std[...,self.target_idx]
#         weight = weight[...,self.target_idx] 

                
#         #compute exact kernel        
#         exp_xc_ = xc*inv_std
#         exp_xt_ = xt*inv_std
#         cos_xc_ = xc*mu
#         cos_xt_ = xt*mu

#         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
#         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
#         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
#         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
        
#         if pi2repuse:
#             Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
#         else:
#             Kxz = weight[None,None,:,:]*torch.exp(-0.5*exp_term )*torch.cos(cos_term)
        
#         #if zitter_flag :
#         likerr = (self.loglik).exp()
#         likerr = torch.clamp(likerr,min=self.loglik_bound[0],max=self.loglik_bound[1])                
#         noise_scale = (zitter+likerr**2)[None,None,None,:]         
#         noise_eye = noise_scale*(torch.eye(ndata).to(xc.device)[None,:,:,None]).repeat(1,1,1,self.num_channels)
#         return Kxz + noise_eye

    
        
#     def eval_Kxz(self,xc,xt=None,zitter=1e-6,zitter_flag = False):
#         """ dependency across channels is considered
#         inputs:
#             xc : (nb,ncontext,ndim,nchannel**2)
#             xt : (nb,ntarget,ndim,nchannel**2)
#         outputs:
#             Kxz : (nb,ntarget,ncontext,nchannel**2)      # assume y-value is 1-d      
#         """
#         #print(len(xc.shape))
#         assert len(xc.shape) == 4         
#         if xt is None:
#             xt = xc
#             zitter_flag = True
                
#         #check size        
#         nb,ndata,ndim,nchannels2 = xc.size()                            
#         #xc_ = xc.repeat(1,1,1,self.num_channels)  #(nb,ndata1,ndim,nchannels**2)
#         #xt_ = xt.repeat(1,1,1,self.num_channels)  #(nb,ndata2,ndim,nchannels**2)

#         mu,inv_std,weight = self.prepare_cross_params()        
#         exp_xc_ = xc*inv_std
#         exp_xt_ = xt*inv_std
#         cos_xc_ = xc*mu
#         cos_xt_ = xt*mu

#         #exp_term = torch.pow(exp_xc_[:,:,None,:]-exp_xt_[:,None,:,:],2).sum(dim=-2)
#         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
#         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
#         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
#         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
        
#         if pi2repuse:        
#             Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
#         else:
#             Kxz = weight[None,None,:,:]*torch.exp(-0.5*exp_term )*torch.cos(cos_term)
        
#         return Kxz

    
    
    
    
#     def eval_Kxz_ind(self,xc,xt=None,zitter=1e-6,zitter_flag = False):
#         """  dependency across channels is not considered
#         inputs:
#             xc : (nb,ncontext,ndim,nchannel)
#             xt : (nb,ntarget,ndim,nchannel)
#         outputs:
#             Kxz : (nb,ntarget,ncontext,nchannel)      # assume y-value is 1-d      
#         """
#         #print(len(xc.shape))
#         assert len(xc.shape) == 4         
#         if xt is None:
#             xt = xc
#             zitter_flag = True
                
#         #check size        
#         nb,ndata,ndim,nchannels2 = xc.size()                            
#         #xc_ = xc.repeat(1,1,1,self.num_channels)  #(nb,ndata1,ndim,nchannels**2)
#         #xt_ = xt.repeat(1,1,1,self.num_channels)  #(nb,ndata2,ndim,nchannels**2)

#         mu,inv_std,weight = self.prepare_cross_params()        
#         mu = mu[...,self.target_idx]
#         inv_std = inv_std[...,self.target_idx]
#         weight = weight[...,self.target_idx] 
        
        
#         exp_xc_ = xc*inv_std
#         exp_xt_ = xt*inv_std
#         cos_xc_ = xc*mu
#         cos_xt_ = xt*mu

#         #exp_term = torch.pow(exp_xc_[:,:,None,:]-exp_xt_[:,None,:,:],2).sum(dim=-2)
#         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
#         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
#         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
#         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
        
#         if pi2repuse:        
#             Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
#         else:
#             Kxz = weight[None,None,:,:]*torch.exp(-0.5*exp_term )*torch.cos(cos_term)
#         return Kxz
        
        
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# #------------------------------------------------------------------------------------------        
        
        
# # class SM_kernel(Multioutput_kernel):
# #     def __init__(self,in_dims=1,num_channels=3, scales=0.1,loglik_err=0.1):
# #         super(SM_kernel,self).__init__(in_dims,num_channels)
                        
# #         self.logsigma = nn.Parameter( torch.log(eps + torch.ones(num_channels)) ,requires_grad=False)         
# #         #self.logmu =    nn.Parameter( torch.log(eps + 0.05 + scales*torch.ones(in_dims,num_channels)) )              #nparams        
# #         self.logmu =    nn.Parameter( torch.log(eps  + 0.0*torch.rand(in_dims,num_channels)) )              #nparams                
# #         self.logstd =   nn.Parameter( torch.log(eps  + scales*torch.ones(in_dims,num_channels)) )    #nparams       
# #         #self.logstd =   nn.Parameter( torch.log(eps + scales*torch.rand(in_dims,num_channels)) )    #nparams                                 
# #         self.loglik =   nn.Parameter( torch.log(eps + loglik_err*torch.ones(num_channels)) )
        
# #         #self.product_list = list(it.product(list(np.arange(self.num_channels)),repeat = 2))
# #         #self.target_idx = [idx  for idx,(ii,jj) in enumerate(self.product_list) if ii==jj]
# #         #self.cross_idx = np.setdiff1d(np.arange(num_channels**2),i_gpsampler.kernel.target_idx).tolist()
    
    
# #     def prepare_cross_params(self,eps=1e-6):
# #         num_channels = self.num_channels
# #         mu = self.logmu.exp()
# #         std = self.logstd.exp()
# #         ndim = self.in_dims

# #         cross_mu,cross_std,cross_weight = [],[],[]
# #         #product_list = list(it.product(list(np.arange(num_channels)),repeat = 2))

# #         for (j,i) in self.product_list:
# #             # std cross
# #             std_j_inv = 1/(std[:,j]+eps)
# #             std_i_inv = 1/(std[:,i]+eps)
# #             std_ji_inv = std_j_inv+std_i_inv
# #             std_ji = 1/(std_ji_inv+eps)

# #             # mu cross            
# #             mu_j = mu[:,j]
# #             mu_i = mu[:,i]
# #             mu_ji = std_ji*(std_j_inv*mu_j + std_i_inv*mu_i)

# #             # normalizer cross                        
# #             std_ji = std[:,j] + std[:,i] 
# #             exp_in = ((mu_j-mu_i)/std_ji)**2            
# #             determinant = (pi2**self.in_dims)*(std_ji**2).prod()
# #             determinant = torch.sqrt(determinant)
# #             weight = (1/determinant)*torch.exp(-0.5*exp_in.sum())
            
# #             cross_mu.append(mu_ji)
# #             cross_std.append(std_ji)
# #             cross_weight.append(weight[None])

# #         cross_mu = torch.cat(cross_mu).reshape(ndim,num_channels**2)
# #         cross_std = torch.cat(cross_std).reshape(ndim,num_channels**2)
# #         cross_weight = torch.cat(cross_weight).reshape(1,num_channels**2)
        
# #         return cross_mu,1/(cross_std+eps),cross_weight
# #         #return cross_mu,cross_std,cross_weight


            

# #     def eval_Kxx(self,xc,zitter=1e-6,zitter_flag = True):
# #         """
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel)
# #         outputs:
# #             Kxz : (nb,ncontext,ncontext,nchannel)      # assume y-value is 1-d      
# #         """
# #         #print(len(xc.shape))
# #         assert len(xc.shape) == 4 and xc.size(-1) == self.num_channels                        
        
# #         #check dataset        
# #         nb,ndata,ndim,nchannels = xc.size()                            
# #         xt = xc
        
# #         #get model parameters
# #         mu,inv_std,weight=self.prepare_cross_params()
# #         mu = mu[...,self.target_idx]
# #         inv_std = inv_std[...,self.target_idx]
# #         weight = weight[...,self.target_idx] 

                
# #         #compute exact kernel        
# #         exp_xc_ = xc*inv_std
# #         exp_xt_ = xt*inv_std
# #         cos_xc_ = xc*mu
# #         cos_xt_ = xt*mu

# #         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
# #         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
# #         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
# #         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
        
# #         if pi2repuse:
# #             Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
# #         else:
# #             Kxz = weight[None,None,:,:]*torch.exp(-0.5*exp_term )*torch.cos(cos_term)

# #         #Kxz = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)

        
# #         #if zitter_flag :
# #         likerr = (2*self.loglik).exp()
# #         noise_scale= (zitter+likerr)[None,None,None,:] 
# #         noise_eye = noise_scale*(torch.eye(ndata).to(xc.device)[None,:,:,None]).repeat(1,1,1,self.num_channels)
# #         return Kxz + noise_eye

    
        
# #     def eval_Kxz(self,xc,xt=None,zitter=1e-6,zitter_flag = False):
# #         """ dependency across channels is considered
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel**2)
# #             xt : (nb,ntarget,ndim,nchannel**2)
# #         outputs:
# #             Kxz : (nb,ntarget,ncontext,nchannel**2)      # assume y-value is 1-d      
# #         """
# #         #print(len(xc.shape))
# #         assert len(xc.shape) == 4         
# #         if xt is None:
# #             xt = xc
# #             zitter_flag = True
                
# #         #check size        
# #         nb,ndata,ndim,nchannels2 = xc.size()                            
# #         #xc_ = xc.repeat(1,1,1,self.num_channels)  #(nb,ndata1,ndim,nchannels**2)
# #         #xt_ = xt.repeat(1,1,1,self.num_channels)  #(nb,ndata2,ndim,nchannels**2)

# #         mu,inv_std,weight = self.prepare_cross_params()        
# #         exp_xc_ = xc*inv_std
# #         exp_xt_ = xt*inv_std
# #         cos_xc_ = xc*mu
# #         cos_xt_ = xt*mu

# #         #exp_term = torch.pow(exp_xc_[:,:,None,:]-exp_xt_[:,None,:,:],2).sum(dim=-2)
# #         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
# #         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
# #         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
# #         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
        
# #         if pi2repuse:        
# #             Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
# #         else:
# #             Kxz = weight[None,None,:,:]*torch.exp(-0.5*exp_term )*torch.cos(cos_term)
        
# #         return Kxz

    
    
    
    
# #     def eval_Kxz_ind(self,xc,xt=None,zitter=1e-6,zitter_flag = False):
# #         """  dependency across channels is not considered
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel)
# #             xt : (nb,ntarget,ndim,nchannel)
# #         outputs:
# #             Kxz : (nb,ntarget,ncontext,nchannel)      # assume y-value is 1-d      
# #         """
# #         #print(len(xc.shape))
# #         assert len(xc.shape) == 4         
# #         if xt is None:
# #             xt = xc
# #             zitter_flag = True
                
# #         #check size        
# #         nb,ndata,ndim,nchannels2 = xc.size()                            
# #         #xc_ = xc.repeat(1,1,1,self.num_channels)  #(nb,ndata1,ndim,nchannels**2)
# #         #xt_ = xt.repeat(1,1,1,self.num_channels)  #(nb,ndata2,ndim,nchannels**2)

# #         mu,inv_std,weight = self.prepare_cross_params()        
# #         mu = mu[...,self.target_idx]
# #         inv_std = inv_std[...,self.target_idx]
# #         weight = weight[...,self.target_idx] 
        
        
# #         exp_xc_ = xc*inv_std
# #         exp_xt_ = xt*inv_std
# #         cos_xc_ = xc*mu
# #         cos_xt_ = xt*mu

# #         #exp_term = torch.pow(exp_xc_[:,:,None,:]-exp_xt_[:,None,:,:],2).sum(dim=-2)
# #         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
# #         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
# #         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
# #         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
        
# #         if pi2repuse:        
# #             Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
# #         else:
# #             Kxz = weight[None,None,:,:]*torch.exp(-0.5*exp_term )*torch.cos(cos_term)
# #         return Kxz
        
        
        
        
        
        
        
        
        
        
        
        








# #--------------------------------------------------------------------------------------------
        
# # class Multioutput_kernel(nn.Module):
# #     def __init__(self,in_dims=1,num_channels=3):
# #         super(Multioutput_kernel,self).__init__()
# #         self.in_dims = in_dims
# #         self.num_channels = num_channels        
# #         return
                        
# #     def eval_Kxz(x,z=None):
# #         raise NotImplementedError
        

        
# # class RBF_kernel(Multioutput_kernel):
# #     def __init__(self,in_dims=1,num_channels=3, scales=0.1,loglik_err=0.1):
# #         super(Rbf_kernel,self).__init__(in_dims,num_channels)                        
# #         self.logsigma = nn.Parameter( torch.log(eps + torch.ones(num_channels)) ,requires_grad=False)   
# #         self.logmu =    nn.Parameter( torch.log(eps + 0.01 + 0*torch.rand(in_dims,num_channels))  ,requires_grad=False)              #nparams                        
# #         self.logstd =   nn.Parameter( torch.log(eps + scales*torch.ones(in_dims,num_channels)) )    #nparams                         
# #         self.loglik =   nn.Parameter( torch.log(eps + loglik_err*torch.ones(num_channels)) )
    
# #     def prepare_cross_params(self,eps=1e-6):
# #         num_channels = self.num_channels
# #         mu = self.logmu.exp()
# #         std = self.logstd.exp()
# #         ndim = self.in_dims

# #         cross_mu,cross_std,cross_weight = [],[],[]
# #         #product_list = list(it.product(list(np.arange(num_channels)),repeat = 2))

# #         for (j,i) in self.product_list:
# #             # std cross
# #             std_j_inv = 1/(std[:,j]+eps)
# #             std_i_inv = 1/(std[:,i]+eps)
# #             std_ji_inv = std_j_inv+std_i_inv
# #             std_ji = 1/(std_ji_inv+eps)

# #             # mu cross            
# #             mu_j = mu[:,j]
# #             mu_i = mu[:,i]
# #             mu_ji = std_ji*(std_j_inv*mu_j + std_i_inv*mu_i)

# #             # normalizer cross                        
# #             std_ji = std[:,j] + std[:,i] 
# #             exp_in = ((mu_j-mu_i)/std_ji)**2            
# #             determinant = (pi2**self.in_dims)*(std_ji**2).prod()
# #             determinant = torch.sqrt(determinant)
# #             weight = (1/determinant)*torch.exp(-0.5*exp_in.sum())
            
# #             cross_mu.append(mu_ji)
# #             cross_std.append(std_ji)
# #             cross_weight.append(weight[None])

# #         cross_mu = torch.cat(cross_mu).reshape(ndim,num_channels**2)
# #         cross_std = torch.cat(cross_std).reshape(ndim,num_channels**2)
# #         cross_weight = torch.cat(cross_weight).reshape(1,num_channels**2)
        
# #         return cross_mu,1/(cross_std+eps),cross_weight

    
    
# #     def eval_Kxz(xc,xt=None,zitter=1e-5,zitter_flag = False):
# #         """
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel)
# #             xt : (nb,ntarget,ndim,nchannel)
# #         outputs:
# #             Kxz : (nb,ntarget,ncontext,nchannel)      # assume y-value is 1-d      
# #         """
        
# #         assert len(xc.size()) == 4         
# #         if xt is None:
# #             xt = xc
# #             zitter_flag = True
                
# #         #check size        
# #         nb,ndata,ndim,nchannels = xc.size()                    
# #         #compute RBF
# #         sigma2 = (2*self.logsigma).exp()[None,None,None,...] + eps   
# #         #length_scales = self.logstd.exp()[None,None,...] + eps              
        
# #         mu_,inv_std_,weight_=self.prepare_cross_params()
        
        
        
# #         xc_= xc/length_scales
# #         xt_= xt/length_scales                
        
# #         xc2 = (xc_**2).sum(dim=-2).unsqueeze(-2)
# #         xt2 = (xt_**2).sum(dim=-2).unsqueeze(-2)        
# #         xxT = xc2 + xt2.permute(0,2,1,3)  -2*torch.einsum('bnmc,bmkc->bnkc',xc_,xt_.permute(0,2,1,3))
# #         Kxz = sigma2*torch.exp(-0.5*xxT) 
# #         if zitter_flag :
# #             likerr = (2*self.loglik).exp()
# #             noise_scale= (zitter+likerr)[None,None,None,:] 
# #             noise_eye = noise_scale*(torch.eye(ndata).to(xc.device)[None,:,:,None]).repeat(1,1,1,self.num_channels)
# #             return Kxz + noise_eye
# #         else:
# #             return Kxz
        
        
               
        
        
# # class SM_kernel(Multioutput_kernel):
# #     def __init__(self,in_dims=1,num_channels=3, scales=0.1,loglik_err=0.1):
# #         super(SM_kernel,self).__init__(in_dims,num_channels)
                        
# #         self.logsigma = nn.Parameter( torch.log(eps + torch.ones(num_channels)) ,requires_grad=False)         
# #         #self.logmu =    nn.Parameter( torch.log(eps + 0.05 + scales*torch.ones(in_dims,num_channels)) )              #nparams        
# #         self.logmu =    nn.Parameter( torch.log(eps  + 0.0*torch.rand(in_dims,num_channels)) )              #nparams                
# #         self.logstd =   nn.Parameter( torch.log(eps + scales*torch.ones(in_dims,num_channels)) )    #nparams       
# #         #self.logstd =   nn.Parameter( torch.log(eps + 0.05 + 0.05*scales*torch.rand(in_dims,num_channels)) )    #nparams                                 
# #         self.loglik =   nn.Parameter( torch.log(eps + loglik_err*torch.ones(num_channels)) )
        
# #         self.product_list = list(it.product(list(np.arange(self.num_channels)),repeat = 2))
# #         self.target_idx = [idx  for idx,(ii,jj) in enumerate(self.product_list) if ii==jj]
        
    
    
# #     def prepare_cross_params(self,eps=1e-6):
# #         num_channels = self.num_channels
# #         mu = self.logmu.exp()
# #         std = self.logstd.exp()
# #         ndim = self.in_dims

# #         cross_mu,cross_std,cross_weight = [],[],[]
# #         #product_list = list(it.product(list(np.arange(num_channels)),repeat = 2))

# #         for (j,i) in self.product_list:
# #             # std cross
# #             std_j_inv = 1/(std[:,j]+eps)
# #             std_i_inv = 1/(std[:,i]+eps)
# #             std_ji_inv = std_j_inv+std_i_inv
# #             std_ji = 1/(std_ji_inv+eps)

# #             # mu cross            
# #             mu_j = mu[:,j]
# #             mu_i = mu[:,i]
# #             mu_ji = std_ji*(std_j_inv*mu_j + std_i_inv*mu_i)

# #             # normalizer cross                        
# #             std_ji = std[:,j] + std[:,i] 
# #             exp_in = ((mu_j-mu_i)/std_ji)**2            
# #             determinant = (pi2**self.in_dims)*(std_ji**2).prod()
# #             determinant = torch.sqrt(determinant)
# #             weight = (1/determinant)*torch.exp(-0.5*exp_in.sum())
            
# #             cross_mu.append(mu_ji)
# #             cross_std.append(std_ji)
# #             cross_weight.append(weight[None])

# #         cross_mu = torch.cat(cross_mu).reshape(ndim,num_channels**2)
# #         cross_std = torch.cat(cross_std).reshape(ndim,num_channels**2)
# #         cross_weight = torch.cat(cross_weight).reshape(1,num_channels**2)
        
# #         return cross_mu,1/(cross_std+eps),cross_weight
# #         #return cross_mu,cross_std,cross_weight


            

# #     def eval_Kxx(self,xc,zitter=1e-6,zitter_flag = True):
# #         """
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel)
# #         outputs:
# #             Kxz : (nb,ncontext,ncontext,nchannel)      # assume y-value is 1-d      
# #         """
# #         #print(len(xc.shape))
# #         assert len(xc.shape) == 4 and xc.size(-1) == self.num_channels                        
        
# #         #check dataset        
# #         nb,ndata,ndim,nchannels = xc.size()                            
# #         xt = xc
        
# #         #get model parameters
# #         mu,inv_std,weight=self.prepare_cross_params()
# #         mu = mu[...,self.target_idx]
# #         inv_std = inv_std[...,self.target_idx]
# #         weight = weight[...,self.target_idx] 

                
# #         #compute exact kernel        
# #         exp_xc_ = xc*inv_std
# #         exp_xt_ = xt*inv_std
# #         cos_xc_ = xc*mu
# #         cos_xt_ = xt*mu

# #         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
# #         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
# #         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
# #         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
# #         Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
# #         #Kxz = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)

        
# #         #if zitter_flag :
# #         likerr = (2*self.loglik).exp()
# #         noise_scale= (zitter+likerr)[None,None,None,:] 
# #         noise_eye = noise_scale*(torch.eye(ndata).to(xc.device)[None,:,:,None]).repeat(1,1,1,self.num_channels)
# #         return Kxz + noise_eye

    
        
# #     def eval_Kxz(self,xc,xt=None,zitter=1e-6,zitter_flag = False):
# #         """ dependency across channels is considered
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel**2)
# #             xt : (nb,ntarget,ndim,nchannel**2)
# #         outputs:
# #             Kxz : (nb,ntarget,ncontext,nchannel**2)      # assume y-value is 1-d      
# #         """
# #         #print(len(xc.shape))
# #         assert len(xc.shape) == 4         
# #         if xt is None:
# #             xt = xc
# #             zitter_flag = True
                
# #         #check size        
# #         nb,ndata,ndim,nchannels2 = xc.size()                            
# #         #xc_ = xc.repeat(1,1,1,self.num_channels)  #(nb,ndata1,ndim,nchannels**2)
# #         #xt_ = xt.repeat(1,1,1,self.num_channels)  #(nb,ndata2,ndim,nchannels**2)

# #         mu,inv_std,weight = self.prepare_cross_params()        
# #         exp_xc_ = xc*inv_std
# #         exp_xt_ = xt*inv_std
# #         cos_xc_ = xc*mu
# #         cos_xt_ = xt*mu

# #         #exp_term = torch.pow(exp_xc_[:,:,None,:]-exp_xt_[:,None,:,:],2).sum(dim=-2)
# #         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
# #         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
# #         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
# #         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
# #         Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)
# #         return Kxz
# #         #if zitter_flag :
# #         #    likerr = (2*self.loglik).exp()
# #         #    noise_scale= (zitter+likerr)[None,None,None,:] 
# #         #    noise_eye = noise_scale*(torch.eye(ndata).to(xc.device)[None,:,:,None]).repeat(1,1,1,nchannels2)
# #         #    return Kxz + noise_eye
# #         #else:
# #         #    return Kxz
        
# #     def eval_Kxz_ind(self,xc,xt=None,zitter=1e-6,zitter_flag = False):
# #         """  dependency across channels is not considered
# #         inputs:
# #             xc : (nb,ncontext,ndim,nchannel)
# #             xt : (nb,ntarget,ndim,nchannel)
# #         outputs:
# #             Kxz : (nb,ntarget,ncontext,nchannel)      # assume y-value is 1-d      
# #         """
# #         #print(len(xc.shape))
# #         assert len(xc.shape) == 4         
# #         if xt is None:
# #             xt = xc
# #             zitter_flag = True
                
# #         #check size        
# #         nb,ndata,ndim,nchannels2 = xc.size()                            
# #         #xc_ = xc.repeat(1,1,1,self.num_channels)  #(nb,ndata1,ndim,nchannels**2)
# #         #xt_ = xt.repeat(1,1,1,self.num_channels)  #(nb,ndata2,ndim,nchannels**2)

# #         mu,inv_std,weight = self.prepare_cross_params()        
# #         mu = mu[...,self.target_idx]
# #         inv_std = inv_std[...,self.target_idx]
# #         weight = weight[...,self.target_idx] 
        
        
# #         exp_xc_ = xc*inv_std
# #         exp_xt_ = xt*inv_std
# #         cos_xc_ = xc*mu
# #         cos_xt_ = xt*mu

# #         #exp_term = torch.pow(exp_xc_[:,:,None,:]-exp_xt_[:,None,:,:],2).sum(dim=-2)
# #         exp_xc2_ = torch.pow(exp_xc_[:,:,None,:],2).sum(dim=-2) 
# #         exp_xt2_ = torch.pow(exp_xt_[:,None,:,:],2).sum(dim=-2) 
# #         exp_term = exp_xc2_ + exp_xt2_  -2*torch.einsum('bijc,bjkc->bikc',exp_xc_,exp_xt_.permute(0,2,1,3))        
# #         cos_term = cos_xc_.sum(dim=-2)[:,:,None,:] - cos_xt_.sum(dim=-2)[:,None,:,:] 
# #         Kxz = weight[None,None,:,:]*torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term)

# #         return Kxz
        
        