from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   

from convcnp.utils import init_sequential_weights, to_multiple        
from test_kernels import SM_kernel        
#from test_gpsampler import Independent_GPsampler,Multioutput_GPsampler                
from test_gpsampler import Independent_GPsampler,Multioutput_GPsampler        
from test_gpsampler import Independent_GPsampler_Proxi


from test_cnnmodels import get_cnnmodels


def collapse_z_samples(t):
    """Merge n_z_samples and batch_size in a single dimension."""
    n_z_samples, batch_size, *rest = t.shape
    return t.contiguous().view(n_z_samples * batch_size, *rest)


def replicate_z_samples(t, n_z_samples):
    """Replicates a tensor `n_z_samples` times on a new first dim."""
    nb,*nleft = t.shape
    t.unsqueeze_(dim=1)
    return t.expand(nb, n_z_samples, *nleft)
    #return t.repeat(1, n_z_samples, *nleft)


eps=1e-6    
num_basis = 5
num_fourierbasis = 10
loglik_err = 0.1

exact_sampler = False
class ICGP_Convnp(nn.Module):
    def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
                      cnntype='shallow', num_postsamples=10,init_lengthscale=0.1):
        super(ICGP_Convnp,self).__init__()
        
        self.modelname = 'gpind'       
        self.in_dims = in_dims
        self.out_dims = out_dims       
        self.num_channels = num_channels
        self.num_samples = num_postsamples
        
        self.cnntype = cnntype
        self.cnn = get_cnnmodels(cnntype)
        
        kernel = SM_kernel(in_dims=in_dims,num_channels=num_channels,scales=init_lengthscale,loglik_err=loglik_err)
        
        #sampler=''
        if exact_sampler==True:
            self.samplertype='exact'
            self.gpsampler = Independent_GPsampler(kernel, 
                                                   in_dims=in_dims,
                                                   out_dims=out_dims,
                                                   num_channels=num_channels,
                                                   num_fourierbasis = num_fourierbasis,
                                                   points_per_unit=self.cnn.points_per_unit,
                                                   multiplier=self.cnn.multiplier)

        else:
            self.samplertype='proxi'
            self.gpsampler =  Independent_GPsampler_Proxi( kernel, 
                                                           in_dims=in_dims,
                                                           out_dims=out_dims,
                                                           num_channels=num_channels,
                                                           num_fourierbasis = num_fourierbasis,
                                                           points_per_unit=self.cnn.points_per_unit,
                                                           multiplier=self.cnn.multiplier)

        
        
        
        gp_linear = nn.Sequential(nn.Linear(2*self.num_channels,8))
        self.gp_linear = init_sequential_weights(gp_linear)


        #num_base = 5        
        self.num_basis = num_basis
        self.num_features = num_channels*num_basis
        cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_features))              
        self.cnn_linear = init_sequential_weights(cnn_linear)
        
        
        self.smoother = ConvDeepset(in_dims=self.in_dims,out_dims=self.out_dims,num_basis=self.num_basis,num_channels=num_channels)
        pred_linear = nn.Sequential(nn.Linear(self.num_features,2*self.num_channels))
        self.pred_linear = init_sequential_weights(pred_linear)
        


    def forward(self,xc,yc,xt,yt=None):        

        nb,ndata,ndim,nchannel = xc.size()
        _ ,ndata2,_,_ = xt.size()
                
        xa_samples,post_samples,density,yc_samples = self.gpsampler.sample_posterior(xc,yc,xt,reorder=False,numsamples=self.num_samples)       
        density_samples = replicate_z_samples(density,self.num_samples)       
        features = torch.cat([post_samples,density_samples],dim=-1)

        features = self.gp_linear(features)
        _,_,ndata,nchannel = features.size()
        features = features.reshape(-1,ndata,nchannel)
        
        features_update = self.cnn(features.permute(0,2,1))
        features_update = self.cnn_linear(features_update.permute(0,2,1))
        features_update = features_update.reshape(nb,self.num_samples,ndata,self.num_basis,self.num_channels)
                
        xt = replicate_z_samples(xt,self.num_samples)        
        xa_samples = replicate_z_samples(xa_samples,self.num_samples)
        xa_samples = xa_samples.unsqueeze(-1).repeat(1,1,1,1,self.num_channels)
        
        #print('features_update.size(),xa_samples.size(),xt.size()')        
        #print(features_update.size(),xa_samples.size(),xt.size())
        
        xt = collapse_z_samples(xt)
        xa_samples = collapse_z_samples(xa_samples)       
        features_update = collapse_z_samples(features_update)
        
        #print('xa_samples.shape,features_update.shape,xt.shape')
        #print( xa_samples.shape,features_update.shape,xt.shape)
        
        #smooth feature
        smoothed_features_update = self.smoother(xa_samples,features_update,xt )              
        smoothed_features_update = smoothed_features_update.reshape(nb,self.num_samples,ndata2,-1)
        smoothed_features_update = smoothed_features_update.permute(1,0,2,3)
        
        #predict        
        features_out = self.pred_linear(smoothed_features_update)                
        pmu,plogstd = features_out.split((self.num_channels,self.num_channels),dim=-1)            
    
        return pmu, 0.01+0.99*F.softplus(plogstd)
        
    
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])

    def compute_regloss_terms(self):
        regtotal = self.gpsampler.regloss
        return regtotal

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
class ConvDeepset(nn.Module):
    #def __init__(self,in_channels=1,out_channles=1):
    def __init__(self,in_dims=1,out_dims=1,num_channels=3,num_basis=5,length_scales=0.1,min_length_scales=1e-6):
        
        super(ConvDeepset,self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_channels = num_channels
        self.num_basis = num_basis        
        #self.log_std = nn.Parameter( torch.log(min_length_scales + length_scales*torch.ones(in_dims,num_channels,num_basis)) )    
        self.log_std = nn.Parameter( torch.log(min_length_scales + length_scales*torch.ones(in_dims,num_basis,num_channels)) )    
        
    
        
    # in : [64, 320, 1, 3]), [64, 47, 1, 3] ,nbasis=5 
    # out :[64, 320,47,5,3]
    def compute_rbf(self,x1,x2=None):
        if x2 is None:
            x2 = x1            
        
        nbatch,npoints,ndim,nchannel = x1.size()
        x1 = x1.unsqueeze(dim=-2)
        x2 = x2.unsqueeze(dim=-2)        
        x1 = x1.repeat(1,1,1,self.num_basis,1)
        x2 = x2.repeat(1,1,1,self.num_basis,1)
        
        std = self.log_std.exp()[None,None,...]         
        x1 = x1/(std+eps)   #(nb,ndata1,ndim,nbasis,nchannel)
        x2 = x2/(std+eps)   #(nb,ndata2,ndim,nbasis,nchannel)

        square_term = (x1**2).sum(dim=2).unsqueeze(dim=2) + (x2**2).sum(dim=2).unsqueeze(dim=1)
        product_term = torch.einsum('bnmjl,bmkjl->bnkjl',x1,x2.permute(0,2,1,3,4))
        dist_term = square_term -2*product_term        
        return torch.exp(-0.5*dist_term) 

    
    # features : [64, 320,1,3]
    # outs :     [64,,47,5,3]    
    def forward(self,xa,features,xt):
        """
        """
        nb,ndata1,ndim,nchannel = xa.size()
        _ ,ndata2, _  , _ = xt.size()        
        wt = self.compute_rbf(xa,xt)      #   [64, 320,47,5,3]
        smoothed_features = (features[:,:,None,:]*wt).sum(dim=1)       
        return smoothed_features 

    
    
    def extra_repr(self):
        line = 'C_in={}, C_out={}, '.format(self.in_dims, self.out_dims)
        #line += 'coords_dim={}, nbhd={}, sampling_fraction={}, mean={}'.format(self.coords_dim, self.num_nbhd, self.sampling_fraction, self.mean)
        return line

    
    
    
    
    
    
    
# class ConvDeepset(nn.Module):
#     #def __init__(self,in_channels=1,out_channles=1):
#     def __init__(self,in_dims=1,out_dims=1,num_channels=3,num_basis=5,length_scales=0.1,min_length_scales=1e-6):
        
#         super(ConvDeepset,self).__init__()
#         self.in_dims = in_dims
#         self.out_dims = out_dims
#         self.num_channels = num_channels
#         self.log_std = nn.Parameter( torch.log(min_length_scales + length_scales*torch.ones(in_dims,num_channels,num_basis)) )    
#         self.num_basis = num_basis
    
#     def compute_dists(self,x1_n,x2_n):
#         dist1 = (x1_n**2).sum(dim=2,keepdim=True) + (x2_n**2).sum(dim=2,keepdim=True).permute(0,2,1,3,4)
#         dist2 = -2*torch.einsum('bnmjl,bmkjl->bnkjl',x1_n,x2_n.permute(0,2,1,3,4))
#         dist = dist1+dist2
#         wt = torch.exp(-0.5*dist)   
#         return wt        
        

#     def compute_rbf(self,x1,x2=None):
#         if x2 is None:
#             x2 = x1            
        
#         nbatch,npoints,ndim,nchannel = x1.size()
#         x1.unsqueeze_(-1)
#         x2.unsqueeze_(-1)
#         x1 = x1.repeat(1,1,1,1,self.num_basis)
#         x2 = x2.repeat(1,1,1,1,self.num_basis)
                        
#         param = self.log_std.exp() + eps        
#         x1_ = x1/(param[None,None,...])
#         x2_ = x2/(param[None,None,...])               
#         return self.compute_dists(x1_,x2_)
        

#     def forward(self,xa,features,xt):
#         """
#         """
#         nb,ndata1,ndim,nchannel = xa.size()
#         _ ,ndata2, _  , _ = xt.size()        
#         wt = self.compute_rbf(xa,xt)         
#         wt = wt.reshape(nb,ndata1,ndata2,-1)
#         smoothed_features = (features[:,:,None,:]*wt).sum(dim=1)       
#         return smoothed_features

    
    
#     def extra_repr(self):
#         line = 'C_in={}, C_out={}, '.format(self.in_dims, self.out_dims)
#         #line += 'coords_dim={}, nbhd={}, sampling_fraction={}, mean={}'.format(self.coords_dim, self.num_nbhd, self.sampling_fraction, self.mean)
#         return line
        

        
    
reglamda = 1.
def compute_loss_gp( pred_mu,pred_std, target_y, z_samples=None, qz_c=None, qz_ct=None):
    
    """
    compute NLLLossLNPF
    # computes approximate LL in a numerically stable way
    # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
    # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
    # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
    # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
    # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)
    
    """
    
    def sum_from_nth_dim(t, dim):
        """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
        return t.view(*t.shape[:dim], -1).sum(-1)


    def sum_log_prob(prob, sample):
        """Compute log probability then sum all but the z_samples and batch."""    
        log_p = prob.log_prob(sample)          # size = [n_z_samples, batch_size, *]    
        sum_log_p = sum_from_nth_dim(log_p, 2) # size = [n_z_samples, batch_size]
        return sum_log_p

    
    p_yCc = Normal(loc=pred_mu, scale=pred_std)    
    if qz_c is not None:
        qz_c = Normal(loc=qz_c[0], scale=qz_c[1])
        
    if qz_ct is not None:
        qz_ct = Normal(loc=qz_ct[0], scale=qz_ct[1])
        
        
    n_z_samples, batch_size, *n_trgt = p_yCc.batch_shape    
    # \sum_t log p(y^t|z). size = [n_z_samples, batch_size]
    sum_log_p_yCz = sum_log_prob(p_yCc, target_y)

    
    # uses importance sampling weights if necessary
    if z_samples is not None:
        # All latents are treated as independent. size = [n_z_samples, batch_size]
        sum_log_qz_c = sum_log_prob(qz_c, z_samples)
        sum_log_qz_ct = sum_log_prob(qz_ct, z_samples)
        # importance sampling : multiply \prod_t p(y^t|z)) by q(z|y_cntxt) / q(z|y_cntxt, y_trgt)
        # i.e. add log q(z|y_cntxt) - log q(z|y_cntxt, y_trgt)
        #print(sum_log_p_yCz, sum_log_qz_c, sum_log_qz_ct)
        sum_log_w_k = sum_log_p_yCz + sum_log_qz_c - sum_log_qz_ct
    else:
        sum_log_w_k = sum_log_p_yCz

    # log_sum_exp_z ... . size = [batch_size]
    log_S_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)
    # - log(n_z_samples)
    log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)    

    #print('log_E_z_sum_p_yCz {}'.format(log_E_z_sum_p_yCz.mean().item()))
    # NEGATIVE log likelihood
    #return -log_E_z_sum_p_yCz
    #return -log_E_z_sum_p_yCz.mean()  #averages each loss over batches 

    return -log_E_z_sum_p_yCz.mean()

#     -----------------------
#     reg loss
#     -----------------------    
#     negll_loss =  
    
#     reg_loss = model.compute_regloss_terms()
#     print(negll_loss, reglamda*regloss)
#     return negll_loss                                   #averages each loss over batches     
#     return negll_loss + reglamda*regloss   #averages each loss over batches 









# import numpy as np
# from dataset_multitask_1d import motask_generator

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# nchannels = 3
# in_channels = nchannels

# train_range,test_range = [0,5],[5,10]   
# tasktype = 'sin3'
# testtype = 'extra'
# dep = True
# gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=dep)


# # In[35]:


# lr = 0.001
# weight_decay=1e-4
# model = CGPconvcnp(in_dims=1,out_dims=1,num_channels=3,num_postsamples=10).cuda()
# opt = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)


# # In[36]:


# model.num_features


# # ## load params

# # In[37]:


# # saved_modelparam_path = './{}_tmp.pth'.format(model.modelname)
# # saved_dict = torch.load(saved_modelparam_path)
# # model.load_state_dict(saved_dict['state_dict'])
# # saved_dict['epoch']


# # ## load dataset

# # In[38]:


# i=5

# save_path_set = './syndata_{}/dep{}_{}_{}'.format(tasktype, dep, testtype, i)
# loaded = torch.load(save_path_set + '.db')
# train_set = loaded['train_set']

# xc = train_set['context_x'][0][...,None,:]
# yc = train_set['context_y'][0]
# xt = train_set['target_x'][0][...,None,:]
# yt = train_set['target_y'][0]

# #print(xc.size(),xt.size())


# # In[39]:


# model.eval()
# if model.modelname in ['p','gp']:
#     pmu_xt,pstd_xt = model(xc.cuda(),yc.cuda(),xt.cuda())       
# else:
#     pmu_xt,pstd_xt = model(xc.squeeze().cuda(),yc.cuda(),xt.squeeze().cuda())       
    


# # In[40]:


# #pmu_xt.size(),pstd_xt.size(),yt.size()


# # In[41]:


# compute_loss(pmu_xt,pstd_xt,yt.cuda())


# # In[42]:


# #tensor(2840210.7500, device='cuda:0', grad_fn=<NegBackward0>)


# # In[ ]:





# # In[ ]:





# # In[ ]:





# # In[43]:



# import matplotlib.pyplot as plt
# xlim_=[0,5]
# figsiz_=(15,3*3)
# fig=plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     #plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=3)    
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)                
#     plt.xlim(xlim_)
#     plt.yticks([-5-1,5+1])    
# plt.show()



# import matplotlib.pyplot as plt
# xlim_=[0,5]
# figsiz_=(15,3*3)
# fig=plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     #plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=3)    
#     plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)                
#     plt.xlim(xlim_)
#     plt.yticks([-5-1,5+1])    
# plt.show()




# # In[49]:



# fig=plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=3)    
#     plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)                
    
# #     plt.plot(xt[0,:,0,i].cpu().data.numpy(),pmu_xt[0,:,i].cpu().data.numpy(),color_list[i]+'*-',markersize=10)
# #     plt.plot(xt[0,:,0,i].cpu().data.numpy(),(pmu_xt+2*pstd_xt)[0,:,i].cpu().data.numpy(),color_list[i]+'-')
# #     plt.plot(xt[0,:,0,i].cpu().data.numpy(),(pmu_xt-2*pstd_xt)[0,:,i].cpu().data.numpy(),color_list[i]+'-')

#     for j in range(5):
#         plt.plot(xt[0,:,0,i].cpu().data.numpy(),pmu_xt[0,j,:,i].cpu().data.numpy(),color_list[i]+'*-',markersize=10)
#         plt.plot(xt[0,:,0,i].cpu().data.numpy(),(pmu_xt+2*pstd_xt)[0,j,:,i].cpu().data.numpy(),color_list[i]+'-')
#         plt.plot(xt[0,:,0,i].cpu().data.numpy(),(pmu_xt-2*pstd_xt)[0,j,:,i].cpu().data.numpy(),color_list[i]+'-')

#     plt.xlim(xlim_)
#     plt.yticks([-5-1,5+1])    
# plt.show()


# # In[45]:


# pstd_xt.size()




# numsamples=2
# xa,post_samples,density = model.gpsampler.sample_posterior(xc.cuda(),yc.cuda(),xt.cuda(),numsamples=numsamples,reorder=True)
# post_mean = post_samples.mean(dim=1)
# post_std = post_samples.std(dim=1)

# figsiz_=(15,8)
# plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10) 
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=15)     
#     plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=7)     
    
#     #plt.plot(xa[0,:,0].cpu().data.numpy(),prior_samples[0,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=3)            
    
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),pmu_xt[0,:,i].cpu().data.numpy(),color_list[i]+'s-')
#     plt.xlim(xlim_)
#     plt.yticks([-5-1,5+1])    
# plt.show()


# plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
  
#     plt.plot(xa[0,:,0].cpu().data.numpy(),post_mean[0,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=2)        

# #     for j in range(numsamples):
# #         plt.plot(xa[0,:,0].cpu().data.numpy(),post_samples[0,j,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=2)        
    
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),pmu_xt[0,:,i].cpu().data.numpy(),color_list[i]+'s-')
#     plt.xlim(xlim_)
#     plt.yticks([-5-1,5+1])    
# plt.show()


# # In[47]:


# plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),0*yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
  
#     plt.plot(xa[0,:,0].cpu().data.numpy(),post_std[0,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=5)        
#     #plt.plot(xa[0,:,0].cpu().data.numpy(),density[0,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=2,alpha=0.4)        

# #     for j in range(numsamples):
# #         plt.plot(xa[0,:,0].cpu().data.numpy(),post_samples[0,j,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=2)        
    
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),pmu_xt[0,:,i].cpu().data.numpy(),color_list[i]+'s-')
#     plt.xlim(xlim_)
#     plt.yticks([0,1])    
# plt.show()



# plt.figure(figsize=figsiz_)
# color_list = ['r','b','g']
# for i in range(nchannels):
#     plt.subplot(3,1,i+1)
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
#     plt.plot(xc[0,:,0,i].cpu().data.numpy(),0*yc[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),yt[0,:,i].cpu().data.numpy(),color_list[i]+'o',markersize=10)     
  
#     #plt.plot(xa[0,:,0].cpu().data.numpy(),post_std[0,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=5)        
#     plt.plot(xa[0,:,0].cpu().data.numpy(),density[0,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=2,alpha=0.4)        

# #     for j in range(numsamples):
# #         plt.plot(xa[0,:,0].cpu().data.numpy(),post_samples[0,j,:,i].cpu().data.numpy(),color_list[i]+'-s',markersize=2)        
    
#     #plt.plot(xt[0,:,0,i].cpu().data.numpy(),pmu_xt[0,:,i].cpu().data.numpy(),color_list[i]+'s-')
#     plt.xlim(xlim_)
#     plt.yticks([0,1])    
# plt.show()


# # In[57]:


# model.gpsampler.logmu.exp()


# # In[ ]:




