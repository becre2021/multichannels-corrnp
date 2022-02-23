import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet
import math

from test_baseline import ConvDeepSet
from test_cnnmodels import get_cnnmodels


def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape (batch, n, 1).
        y (tensor): Inputs of shape (batch, m, 1).

    Returns:
        tensor: Pair-wise distances of shape (batch, n, m).
    """
    return (x - y.permute(0, 2, 1)) ** 2



def collapse_z_samples_batch(t):
    """Merge n_z_samples and batch_size in a single dimension."""
    n_z_samples, batch_size, *rest = t.shape
    return t.contiguous().view(n_z_samples * batch_size, *rest)


def extract_z_samples_batch(t, n_z_samples, batch_size):
    """`reverses` collapse_z_samples_batch."""
    _, *rest = t.shape
    return t.view(n_z_samples, batch_size, *rest)


def replicate_z_samples(t, n_z_samples):
    """Replicates a tensor `n_z_samples` times on a new first dim."""
    return t.unsqueeze(0).expand(n_z_samples, *t.shape)






device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pi = math.pi
eps = 1e-6      

class Convcnp_latent(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, in_dims=1,out_dims=1,num_channels=1,cnntype='shallow',init_lengthscale=0.1, nbasis=5,num_postsamples=10):
        
        super(Convcnp_latent, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()                
        
        self.modelname = 'baselatent'        
        self.samplertype='exact'
        
        # Instantiate encoder
        self.in_dims = 1
        self.out_dims = 1
        self.num_channels = num_channels
#         self.encoder = ConvDeepSet(in_channels = self.num_channels,
#                                    out_channels=self.rho.in_channels,
#                                    init_length_scale=init_length_scale)
#         self.rho = rho
#         self.multiplier = 2 ** self.rho.num_halving_layers


        self.encoder = ConvDeepSet(in_channels = self.num_channels,
                                   out_channels= 8,
                                   init_lengthscale=init_lengthscale)

        self.cnntype = cnntype
        self.cnn = get_cnnmodels(cnntype)  #in:8 -> out:8
        #self.cnn = get_cnnmodels('deep') # not compatiable yet
        

        self.num_samples = num_postsamples
        self.nbasis = nbasis 
        self.num_features = self.num_channels*self.nbasis
        self.cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,2*self.num_features))         
        self.smoother =  FinalLayer(in_channels=self.num_channels,
                                    nbasis = self.nbasis,
                                    init_lengthscale=init_lengthscale)
        pred_linear = nn.Sequential(nn.Linear(self.num_channels*self.nbasis,2*self.num_channels))         
        self.pred_linear = init_sequential_weights(pred_linear)
        
        
        

    def compute_xgrid(self,x,y,x_out,x_thres=1.0):
        x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - x_thres
        x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + x_thres        
        num_points = int(to_multiple(self.cnn.points_per_unit * (x_max - x_min),self.cnn.multiplier))             
        x_grid = torch.linspace(x_min , x_max , num_points).to(x.device)
        
        nb,npoints,nchannel = x.size()        
        x_grid = x_grid[None, :, None].repeat(nb, 1, nchannel)
        return x_grid
        
        
        

    def compute_hgrid(self,h):
        h = self.activation(h) 
        h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
        nb,_,ngrid = h.size()    
                
        #h = self.rho(h)
        h = self.cnn(h)        
        h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
        nb,ngrid,_= h_grid.size()
        h_grid = h_grid.reshape(-1,h_grid.size(-1))
        h_grid = self.cnn_linear(h_grid).reshape(nb,ngrid,-1)        
        return h_grid
                
        
    def samples_z(self,h_mu,h_std,num_samples=10):        
        """
        inputs:
            h_mu: (nb,ngrid,num_feature)
            h_std: (nb,ngrid,num_feature)            
        """
        h_mu = h_mu[None,:,:,:]
        h_std = 0.01+0.99*torch.sigmoid(h_std)[None,:,:,:]        
        #eps = torch.randn(num_samples,h_std.size(1),h_std.size(-1))  :  suspect that too volatile 
        eps = torch.randn(num_samples,1,h_std.size(-1))               #: change as 
        eps = Variable(eps[:,:,None,:]).to(h_mu.device)
        
        z_samples =  h_mu + h_std*eps        
        return z_samples,h_mu.squeeze(),h_std.squeeze()

        
        
        
        
    def forward(self, x, y, xout, yout=None):
        """Run the model forward.
        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).
            
        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        nb,ndata,nchannel = x.size()        
        _ ,ndata2,_ = xout.size()
        
        
        xgrid = self.compute_xgrid(x,y,xout)
        concat_nh1h0,nh1,h1,h0 = self.encoder(x,y,xgrid)        
                
        hgrid = self.compute_hgrid(concat_nh1h0)
        
        #h_grid = (...,2d) so that d,d modeled for latent mu and std
        assert hgrid.size(-1) % 2 == 0
        hgrid_mu,hgrid_std = hgrid.split((self.num_features,self.num_features),dim=-1)
        zsamples,hmu_c,hstd_c = self.samples_z(hgrid_mu,hgrid_std,num_samples=self.num_samples)
        zsamples_c = collapse_z_samples_batch(zsamples)
        zsamples_c = zsamples_c.reshape(zsamples_c.size(0),zsamples_c.size(1),self.nbasis,-1)
        
        if yout is not None:
            xgrid_t = self.compute_xgrid(xout,yout,xout)            
            concat_n_h1h0_t,_,_,_ = self.encoder(xout,yout,xgrid_t)        
            hgrid_t = self.compute_hgrid(concat_n_h1h0_t)
            assert h_grid_t.size(-1) % 2 == 0
            hgrid_mu_t,hgrid_std_t = hgrid_t.split((hgrid_t.size(-1),hgrid_t.size(-1)),dim=-1)
            zsamples_t,hmu_t,hstd_t = self.samples_z(hgrid_mu_t,hgrid_std_t,num_samples=self.num_samples)
            
                        
        xgrid = collapse_z_samples_batch(replicate_z_samples(xgrid, n_z_samples=self.num_samples))      
        xout = collapse_z_samples_batch(replicate_z_samples(xout, n_z_samples=self.num_samples))         
        
        # smooth feaute
        smoothed_hout = self.smoother(xgrid,zsamples_c,xout)
        smoothed_hout = smoothed_hout.reshape(nb,self.num_samples,ndata2,-1)
        smoothed_hout = smoothed_hout.permute(1,0,2,3)
        
        # predict
        pred_hout = self.pred_linear(smoothed_hout)        
        pmu,plogstd = pred_hout.split((self.num_channels,self.num_channels),dim=-1)
        
        
        #print('y_mu.shape,y_logstd.shape')
        #print(pmu.shape,plogstd.shape)
        if yout is None:
            return pmu, 0.01+0.99*F.softplus(plogstd)
        else:
            return pmu, 0.01+0.99*F.softplus(plogstd),zsamples_t,(hmu_c,hstd_c),(hmu_t,hstd_t)
    
    
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])


    def compute_regloss_terms(self):
        #regtotal = self.gpsampler.regloss
        return 0.0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
class FinalLayer(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self,  in_channels=1, out_channels = 1, nbasis = 1, init_lengthscale = 1.0,min_init_lengthscale=1e-6):
        super(FinalLayer, self).__init__()
        
        #self.out_channels = in_channels             
        #self.in_channels_hidden = in_channels*self.out_channels       
        self.sigma_fn = torch.exp        
        self.nbasis = nbasis
        self.in_channels = in_channels        
        self.out_channels = out_channels               
                
        #self.g = self.build_weight_model()
        #linear = nn.Sequential(nn.Linear(self.nbasis, self.out_channels))
        #self.g = init_sequential_weights(linear)
        
        self.sigma = nn.Parameter(np.log(min_init_lengthscale+init_lengthscale*torch.ones(self.nbasis,self.in_channels)), requires_grad=True)          
        
        
        
    def compute_rbf(self,x1,x2=None ,eps=eps):
        if x2 is None:
            x2 = x1            
        # Compute shapes.            
        nbatch,npoints,nchannel = x1.size()
        
        #compute rbf over multiple channels
        dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)
        dists = dists.unsqueeze(dim=-2).repeat(1,1,1,self.nbasis,1)        
        scales = self.sigma.exp()
        scales = scales[None, None, None, :,:]  
        
        dists = dists/(scales + eps)
        wt = torch.exp(-0.5*dists**2)   
        return wt
        
        
    #nbasis == 5 case    
    def forward(self, x_grid, h_grid, target_x):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """

        nb,ntarget,nchannel = target_x.size()        
        _,ngrid,_,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
        
        wt = self.compute_rbf(x_grid,target_x)        
        h = h_grid[:,:,None,:,:] #(nb,ngrid,1,nbasis,nchannels)
        h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
        return h_out
    

    
    
        
    
    
    
    
    
    
def compute_loss_baselinelatent( pred_mu, pred_std, target_y, z_samples=None, qz_c=None, qz_ct=None):
    
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
    return -log_E_z_sum_p_yCz.mean()  #averages each loss over batches 

    
    
    
    
    
    
    
    
    
    
    
    

    
###########################################    
# configuration for parameters
#mu_scale = 0.1
#init_length_scale = 0.01 #sin3
#init_length_scale = 0.5 #matern 3


    
    
    
    
    
    
    
    
    


##########################################    
# convset not necessary
##########################################

# import math
# pi = math.pi
# eps = 1e-6
# class ConvDeepSet(nn.Module):
#     """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

#     Args:
#         out_channels (int): Number of output channels.
#         init_length_scale (float): Initial value for the length scale.
#     """

#     def __init__(self, in_channels,out_channels, init_length_scale=1.0,min_init_length_scale=1e-6):
#         super(ConvDeepSet, self).__init__()
#         self.in_channels = in_channels        
#         self.out_channels = out_channels
#         self.g = self.build_weight_model()
        
                
#         self.sigma = nn.Parameter(torch.log(min_init_length_scale+init_length_scale*torch.ones(self.in_channels)), requires_grad=True)        
#         #self.mu = nn.Parameter(np.log(1* torch.rand(self.in_channels)), requires_grad=True)
#         self.sigma_fn = torch.exp

#     def build_weight_model(self):
#         """Returns a point-wise function that transforms the
#         (in_channels + 1)-dimensional representation to dimensionality
#         out_channels.

#         Returns:
#             torch.nn.Module: Linear layer applied point-wise to channels.
#         """
#         #model = nn.Sequential(nn.Linear(self.in_channels, self.out_channels))
#         model = nn.Sequential(nn.Linear(2*self.in_channels, self.out_channels))
        
#         init_sequential_weights(model)
#         return model
    
        
    
#     def compute_rbf(self,x1,x2=None):
#         if x2 is None:
#             x2 = x1            
#         # Compute shapes.            
#         nbatch,npoints,nchannel = x1.size()
        
#         #compute rbf over multiple channels
#         dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)        
#         scales = self.sigma_fn(self.sigma)[None, None, None, :]                
        
#         factors = 1
#         if dists.size(-1) != scales.size(-1):
#             factors = scales.size(-1) // dists.size(-1) 
#             dists = dists.repeat(1,1,1,factors)
#         #print(dists.size(),scales.size())
        
        
#         dists /= (scales + eps)
#         wt = torch.exp(-0.5*dists**2)   
#         return wt,factors
        
#     def forward(self, context_x, context_y, x_grid):
#         """Forward pass through the layer with evaluations at locations t.

#         Args:
#             x (tensor): Inputs of observations of shape (n, 1).
#             y (tensor): Outputs of observations of shape (n, in_channels).
#             t (tensor): Inputs to evaluate function at of shape (m, 1).

#         Returns:
#             tensor: Outputs of evaluated function at z of shape
#                 (m, out_channels).
#         """


#         nbatch,npoints,nchannel = context_x.size()
#         wt,factors = self.compute_rbf(context_x,x_grid)
#         h0 = wt.sum(dim=1)

#         if factors > 1:
#             context_y = context_y.repeat(1,1,factors)
        
#         h1 = (context_y.unsqueeze(dim=-2)*wt).sum(dim=1)
#         n_h1 = h1/(h0+eps)        
        
#         y_out = torch.cat((h0, n_h1), dim=-1)    #(nbatch, ngrid,2*noutchannel)         
#         y_out = y_out.view(-1,2*self.in_channels)  #(nbatch, ngrid,2*noutchannel) 
#         y_out = self.g(y_out)
#         y_out = y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)

#         #return y_out,h1,h0
#         return y_out,n_h1,h1,h0
    
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    model_m = ConvCNP_Multi(rho = UNet(),points_per_unit=128)
