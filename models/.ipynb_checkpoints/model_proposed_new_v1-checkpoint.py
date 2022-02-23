import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#import stheno.torch as stheno

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



    
    
#     def compute_sm(self,x1,x2 = None):
#         if x2 is None:
#             x2 = x1            
#         # Compute shapes.            
#         nbatch,npoints,nchannel = x1.size()
        
#         #compute rbf over multiple channels
#         dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)        
#         mu_ = self.sigma_fn(self.mu)[None, None, None, :]  
#         std_ = self.sigma_fn(self.sigma)[None, None, None, :]  
        
#         #exp_term_dist = dists/std_
#         exp_term_dist = dists*std_        
#         cos_term_dist = dists*mu_
        
#         wt = torch.exp(-2*(pi**2)*(exp_term_dist**2))*torch.cos(2*pi*cos_term_dist) 
#         return wt


###########################################    
# configuration for parameters
#mu_scale = 0.1
#init_length_scale = 0.01 #sin3
#init_length_scale = 0.5 #matern 3

###########################################
 

from torch.fft import rfftn, irfftn,rfft, irfft

def get_convolved_h(h_grid,nchannel=3,p=None):    
    nb,ngrid,nkernels = h_grid.size()
    #permuted_index = np.random.permutation(nkernels)        
    
    #p = np.random.randint(1,nkernels)
    #p=6    
    if p is None:
        p = np.random.randint(1,nchannel)
    permuted_index = np.roll(np.arange(nkernels),-p)
    
    #print(p,np.arange(nkernels),permuted_index)
    return convolve_1d_functional_h(h_grid,h_grid[:,:,permuted_index])



def convolve_1d_functional_h(h_i,h_j,target_dim=1,eps=1e-4):
    """
    inputs
        h_i : (nb,ngrids,nchannel)
        h_j : (nb,ngrids,nchannel)    
    outputs
        h_ij : (nb,ngrids,nchannel)
    """
    
    
    if h_i.size(-1) > h_j.size(-1):
        p2d = (0,h_i.size(-1) - h_j.size(-1))
        h_j = F.pad(h_j, p2d, "constant", 0)
    
    
    
    #nhiddens = h_i.size(1)
    nhiddens = h_i.size(target_dim)
    
    if nhiddens % 2 != 0:
        nhiddens -= 1
        h_i = h_i[:,:-1,:]
        h_j = h_j[:,:-1,:]    
    
    f_h_i = rfft(h_i,dim=target_dim)
    f_h_j = rfft(h_j,dim=target_dim)
    

    #plt.scatter(i,(torch.roll(a,shifts=i,dims=-1)*b).sum(dim=1))
    #f_h_i.imag *= -1    
    #n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim)        
    
    #plt.scatter(i,(a*torch.roll(b,shifts=i,dims=-1)).sum(dim=1))
    f_h_j.imag *= -1    
    n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim)        
    #n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim) / (h_i.norm(dim=-1,keepdim=True)*h_j.norm(dim=-1,keepdim=True) + eps)       
    
    #return n_ij_conv/(n_ij_conv.norm(dim=-1,keepdim=True) + eps)
    #return n_ij_conv/(h_i.norm(dim=-1,keepdim=True)*h_j.norm(dim=-1,keepdim=True) + eps)
    return (n_ij_conv-n_ij_conv.mean(dim=1,keepdim=True))/(n_ij_conv.std(dim=1,keepdim=True) + eps)




import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#import stheno.torch as stheno

import convcnp.data
from convcnp.experiment import report_loss, RunningAverage
from convcnp.utils import gaussian_logpdf, init_sequential_weights, to_multiple
from convcnp.architectures import SimpleConv, UNet
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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




###########################################    
# configuration for parameters
#mu_scale = 0.1
#init_length_scale = 0.01 #sin3
#init_length_scale = 0.5 #matern 3

###########################################
 

from torch.fft import rfftn, irfftn,rfft, irfft

def get_convolved_h(h_grid,nchannel=3,p=None):    
    nb,ngrid,nkernels = h_grid.size()
    #permuted_index = np.random.permutation(nkernels)        
    
    #p = np.random.randint(1,nkernels)
    #p=6    
    if p is None:
        p = np.random.randint(1,nchannel)
    permuted_index = np.roll(np.arange(nkernels),-p)
    
    #print(p,np.arange(nkernels),permuted_index)
    return convolve_1d_functional_h(h_grid,h_grid[:,:,permuted_index])



def convolve_1d_functional_h(h_i,h_j,target_dim=1,eps=1e-4):
    """
    inputs
        h_i : (nb,ngrids,nchannel)
        h_j : (nb,ngrids,nchannel)    
    outputs
        h_ij : (nb,ngrids,nchannel)
    """
    
    
    if h_i.size(-1) > h_j.size(-1):
        p2d = (0,h_i.size(-1) - h_j.size(-1))
        h_j = F.pad(h_j, p2d, "constant", 0)
    
    
    
    #nhiddens = h_i.size(1)
    nhiddens = h_i.size(target_dim)
    
    if nhiddens % 2 != 0:
        nhiddens -= 1
        h_i = h_i[:,:-1,:]
        h_j = h_j[:,:-1,:]    
    
    f_h_i = rfft(h_i,dim=target_dim)
    f_h_j = rfft(h_j,dim=target_dim)
    

    #plt.scatter(i,(torch.roll(a,shifts=i,dims=-1)*b).sum(dim=1))
    #f_h_i.imag *= -1    
    #n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim)        
    
    #plt.scatter(i,(a*torch.roll(b,shifts=i,dims=-1)).sum(dim=1))
    f_h_j.imag *= -1    
    n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim)        
    #n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim) / (h_i.norm(dim=-1,keepdim=True)*h_j.norm(dim=-1,keepdim=True) + eps)       
    
    #return n_ij_conv/(n_ij_conv.norm(dim=-1,keepdim=True) + eps)
    #return n_ij_conv/(h_i.norm(dim=-1,keepdim=True)*h_j.norm(dim=-1,keepdim=True) + eps)
    return (n_ij_conv-n_ij_conv.mean(dim=1,keepdim=True))/(n_ij_conv.std(dim=1,keepdim=True) + eps)


    
    
# ----------------------------------
# model v1
# ----------------------------------


class ConvCNP_Multi_CC(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, in_channels, rho,cc_rho, points_per_unit,rbf_init_l=None, nbasis=5):
        super(ConvCNP_Multi_CC, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.cc_rho = cc_rho
        
        self.multiplier = 2 ** self.rho.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        if rbf_init_l is None:
            init_length_scale = 2.0 / self.points_per_unit            
        else:
            init_length_scale = rbf_init_l
        
        # Instantiate encoder
        self.in_channels = in_channels
        self.encoder = ConvDeepSet_encoder_multi(in_channels = self.in_channels,
                                                 out_channels=self.rho.in_channels,
                                                 init_length_scale=init_length_scale)

#         # -----------
#         # consider multiple encoding features
#         # -----------        
#         self.encoder = ConvDeepSet_multi(in_channels = self.rho.in_channels,
#                                          out_channels = self.rho.out_channels,
#                                          init_length_scale=init_length_scale)

    

        #self.nbasis = 5
        self.nbasis = nbasis 
        linear = nn.Sequential(nn.Linear(self.rho.out_channels,self.in_channels*self.nbasis))        
        self.linear = init_sequential_weights(linear)
        
        self.mean_layer = FinalLayer_multi(in_channels=self.in_channels,
                                           nbasis = self.nbasis,
                                           init_length_scale=init_length_scale)
        
        self.logstd_layer = FinalLayer_multi(in_channels=self.in_channels,
                                             nbasis = self.nbasis,
                                             init_length_scale=init_length_scale)

#         cc_linear = nn.Sequential(nn.Linear(self.cc_rho.out_channels,self.in_channels*self.nbasis))         
#         self.cc_linear = init_sequential_weights(cc_linear)

#         self.cc_mean_layer = FinalLayer_multi(in_channels=self.in_channels,
#                                            nbasis = self.nbasis,
#                                            init_length_scale=init_length_scale)
        
#         self.cc_logstd_layer = FinalLayer_multi(in_channels=self.in_channels,
#                                              nbasis = self.nbasis,
#                                              init_length_scale=init_length_scale)
        
        

                                           
    def compute_xgrid(self,x,y,x_out,x_thres=0.1):
        x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - x_thres
        x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + x_thres
        
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))
        
        if x.is_cuda:
            x_grid = torch.linspace(x_min, x_max, num_points).to(device)
        else:
            x_grid = torch.linspace(x_min, x_max, num_points)
            
        # context
        nb,npoints,nchannel = x.size()        
        x_grid = x_grid[None, :, None].repeat(nb, 1, nchannel)
        return x_grid
        
        
        

    def compute_hgrid(self,h,option = 'base'):
        h = self.activation(h) 
        h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
        nb,_,ngrid = h.size()    
                
        if option == 'base':
            #print('option in base')
            h = self.rho(h)
            h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
            nb,ngrid,_= h_grid.size()
            h_grid = h_grid.reshape(-1,h_grid.size(-1))
            h_grid = self.linear(h_grid).reshape(nb,ngrid,self.nbasis,-1)        

        if option == 'cross':
            #print('option in cross')            
            h = self.cc_rho(h)
            h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
            nb,ngrid,_= h_grid.size()
            h_grid = h_grid.reshape(-1,h_grid.size(-1))
            h_grid = self.cc_linear(h_grid).reshape(nb,ngrid,self.nbasis,-1)        
            
        return h_grid
                

        
    def forward(self, x, y, x_out, seperate=False):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).
            
        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        nb,npoints,nchannel = x.size()        
        x_grid = self.compute_xgrid(x,y,x_out)
        #concat_n_h1h0,n_h1,h1,h0 = self.encoder(x,y,x_grid)        
        y_out,n_h1,h1,h0 = self.encoder(x,y,x_grid)        

        
        #----------------
        #obs
        #----------------        
        concat_n_h1h0 = y_out[0]
        h_grid = self.compute_hgrid(concat_n_h1h0,option='base')
        mean = self.mean_layer(x_grid,h_grid,x_out,option='mean')
        std = self.sigma_fn(self.logstd_layer(x_grid,h_grid,x_out,option='std'))   
        #return mean, std
        

#         # ----------------
#         #  cross
#         # ----------------        
#         cc_concat_n_h1h0 = y_out[1]
#         cc_h_grid = self.compute_hgrid(cc_concat_n_h1h0 ,option='cross')                    
#         cc_mean = self.cc_mean_layer(x_grid,cc_h_grid,x_out)
#         cc_std = self.sigma_fn(self.cc_logstd_layer(x_grid,cc_h_grid,x_out))   
                        
        if seperate:
            return (mean,cc_mean),(std,cc_std)
        else:
            #return mean+cc_mean,std+cc_std
            #return cc_mean,cc_std
            return mean,std
        
        
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])


    
    
    
# -----------------------------
# conv random fourier 
# -----------------------------
from torch.autograd import Variable
import math
pi2 = 2*math.pi


class conv_random_fourier(nn.Module):    
    def __init__(self,nbasis=10,in_channels=3,w_mu= None,w_std = None):
        super(conv_random_fourier, self).__init__()
        
        self.in_channels = in_channels       
        self.nbasis = nbasis
 
        if w_mu is None:
            w_mu = torch.tensor(0).float()
        if w_std is None:
            w_std = torch.tensor(1).float()
        
        self.w_mu = nn.Parameter(w_mu)
        self.w_std = nn.Parameter(w_std)         
        self.conv = nn.Conv1d(in_channels=self.in_channels, 
                          out_channels=self.in_channels,
                          kernel_size=5, 
                          stride=1,
                          padding=2, 
                          groups=in_channels)        
        
#         self.conv2 = nn.Conv1d(in_channels=self.in_channels, 
#                           out_channels=self.in_channels,
#                           kernel_size=7, 
#                           stride=1,
#                           padding=3, 
#                           groups=in_channels)        
        
#         self.conv3 = nn.Conv1d(in_channels=self.in_channels, 
#                           out_channels=self.in_channels,
#                           kernel_size=15, 
#                           stride=1,
#                           padding=7, 
#                           groups=in_channels)        
        
        
#         self.conv = nn.Conv1d(in_channels=self.in_channels, 
#                           out_channels=self.in_channels,
#                           kernel_size=5, 
#                           stride=1,
#                           padding=2, 
#                           groups=1)        
        
        self.batchnorm = nn.BatchNorm1d(self.in_channels)
        #self.batchnorm2 = nn.BatchNorm1d(self.in_channels)
        #self.batchnorm3 = nn.BatchNorm1d(self.in_channels)

        
        return 
        
    def sample_w_b(self,nb):    
        eps1 = Variable(torch.randn(nb,self.nbasis),requires_grad=False)
        eps2 = Variable(pi2*torch.rand(nb,self.nbasis),requires_grad=False)
        
        if self.w_mu.is_cuda :
            sample_w = self.w_mu + self.w_std*eps1.cuda()
            sample_b = eps2.cuda()
        else:
            sample_w = self.w_mu + self.w_std*eps1
            sample_b = eps2
            
        #sample_w = sample_w[None,None,:,None]
        #sample_b = sample_b[None,None,:,None]        
        sample_w = sample_w[:,None,:,None]
        sample_b = sample_b[:,None,:,None]                
        return sample_w,sample_b
        
        
    def forward(self,x_grid):
        sample_w,sample_b = self.sample_w_b(x_grid.size(0))        
        #print(sample_w.size(),sample_b.size())
        #print(x_grid.size())
        if x_grid.dim() == 3:
            x_grid = x_grid[:,:,None,:]
        x_grid = x_grid.repeat(1,1,self.nbasis,1)
        
        
       # print(x_grid.size(),sample_w.size(),sample_b.size())        
        Psi = torch.cos(pi2*sample_w*x_grid+sample_b)
        if x_grid.is_cuda : 
            random_w =torch.randn(1,1,self.nbasis,1).cuda()
        else:
            random_w =torch.randn(1,1,self.nbasis,1)
             
        Psi = np.sqrt(2/self.nbasis)*(Psi*random_w).sum(dim=-2)        
        
        Psi1 = self.conv(Psi.permute(0,2,1))
        #print(Psi1.size())
        Psi1 = self.batchnorm(Psi1)
        Psi1 = Psi1.permute(0,2,1)
        return Psi
        
#         Psi2 = self.conv(Psi.permute(0,2,1))
#         #print(Psi2.size())        
#         Psi2 = self.batchnorm2(Psi2)
#         Psi2 = Psi2.permute(0,2,1)

#         Psi3 = self.conv(Psi.permute(0,2,1))
#         #print(Psi3.size())        
#         Psi3 = self.batchnorm3(Psi3)
#         Psi3 = Psi3.permute(0,2,1)
        
        #Psi = torch.relu(Psi.permute(0,2,1))        
#        return Psi1
        #return Psi1 + Psi2 + Psi3
        
        
    
    

###########################################    
# encoder and decoder
###########################################

import math
pi = math.pi

eps = 1e-6
class ConvDeepSet_encoder_multi(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels,out_channels, init_length_scale=1.0,min_init_length_scale=1e-6):
        super(ConvDeepSet_encoder_multi, self).__init__()
        self.in_channels = in_channels        
        self.out_channels = out_channels
        #self.g = self.build_weight_model()
        self.g,self.cc_g = self.build_weight_model()        
                
        self.sigma = nn.Parameter(torch.log(min_init_length_scale+init_length_scale*torch.ones(self.in_channels)), requires_grad=True)        
        #self.mu = nn.Parameter(np.log(1* torch.rand(self.in_channels)), requires_grad=True)
        self.sigma_fn = torch.exp
        
        
        # ---------------
        # fourier basis
        self.nbasis_fourier = 10
        self.fourier_basis = conv_random_fourier(nbasis=self.nbasis_fourier,
                                                 in_channels=self.in_channels)
        
    
    
        
        

    def build_weight_model(self):
        """Returns a point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        #model = nn.Sequential(nn.Linear(self.in_channels, self.out_channels))
        self.tmp_channel = 2*self.in_channels
        #self.tmp_channel = 4*self.in_channels        
        model = nn.Sequential(nn.Linear(self.tmp_channel, self.out_channels))        
        init_sequential_weights(model)
        #return model
    
        self.cc_tmp_channel = 3*self.in_channels
        #self.tmp_channel = 4*self.in_channels        
        model2 = nn.Sequential(nn.Linear(self.cc_tmp_channel, self.out_channels))

        return model,model2
    
    
    
    
    def compute_rbf(self,x1,x2=None):
        if x2 is None:
            x2 = x1            
        # Compute shapes.            
        nbatch,npoints,nchannel = x1.size()
        
        #compute rbf over multiple channels
        dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)        
        scales = self.sigma_fn(self.sigma)[None, None, None, :]                
        
        factors = 1
        if dists.size(-1) != scales.size(-1):
            factors = scales.size(-1) // dists.size(-1) 
            dists = dists.repeat(1,1,1,factors)
        #print(dists.size(),scales.size())
        
        
        dists /= (scales + eps)
        wt = torch.exp(-0.5*dists**2)   
        return wt,factors

    
    

    def forward(self, contex_x,context_y,x_grid ):
        # old code
        # fourier basis, but no correlation between each featuresw
        #return self.forward_v1(contex_x,context_y,x_grid )
        
        return self.forward_v2(contex_x,context_y,x_grid )
        
        
    def forward_v1(self, context_x, context_y, x_grid):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """


        nbatch,npoints,nchannel = context_x.size()
        wt,factors = self.compute_rbf(context_x,x_grid)
        h0 = wt.sum(dim=1)

        if factors > 1:
            context_y = context_y.repeat(1,1,factors)
        
        h1 = (context_y.unsqueeze(dim=-2)*wt).sum(dim=1)
        n_h1 = h1/(h0+eps)        
   

        # ------------------------
        # nfourier basis
        # ------------------------        
        grid_prior = self.fourier_basis(x_grid)
        #print(grid_prior.size())
        n_h1 = n_h1 + grid_prior
        
        




        y_out = torch.cat((h0, n_h1), dim=-1)    #(nbatch, ngrid,2*noutchannel)         
        #print('y_out {}'.format(y_out.size()))        
        y_out = y_out.view(-1,self.tmp_channel)  #(nbatch, ngrid,2*noutchannel) 
        y_out = self.g(y_out)
        y_out = y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)
        #return y_out,n_h1,h1,h0

    
        # -------------------
        # cross features 
        # -------------------
        cc_n_h1 = []
        for j in range(1,self.in_channels):
            #cc_n_h1.append(get_convolved_h(n_h1,nchannel=self.in_channels,p=1)[:,:,None,:])
            cc_n_h1.append(get_convolved_h(n_h1,nchannel=self.in_channels,p=j))            
        cc_n_h1 = torch.cat(cc_n_h1,dim=-1)   # [1-2,2-3,3-1,1-3,2-1,3-2]^T 
                                              # or
                                              # [1-2,2-3,3-1,   .reshape(nb,hgrid,-1,self.in_channel)                 
                                              #  1-3,2-1,3-2]   

        cc_y_out = torch.cat((h0, cc_n_h1), dim=-1)    #(nbatch, ngrid,2*noutchannel)        
        
        #-------------
        #detach()
        #cc_y_out = cc_y_out.detach().clone()
        #-------------
        
        
        #print('cc_y_out {}'.format(cc_y_out.size()))
        cc_y_out = cc_y_out.view(-1,self.cc_tmp_channel)  #(nbatch, ngrid,2*noutchannel) 
        cc_y_out = self.cc_g(y_out)
        cc_y_out = cc_y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)        
        
        #print('y_out {}, cc_y_out {}'.format(y_out.size(),cc_y_out.size()))
        #print('')
        return (y_out,cc_y_out),n_h1,h1,h0
    


    
        
    def forward_v2(self, context_x, context_y, x_grid):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """


        nbatch,npoints,nchannel = context_x.size()
        wt,factors = self.compute_rbf(context_x,x_grid)
        h0 = wt.sum(dim=1)

        if factors > 1:
            context_y = context_y.repeat(1,1,factors)
        
        h1 = (context_y.unsqueeze(dim=-2)*wt).sum(dim=1)
        n_h1 = h1/(h0+eps)        
   

#         # ------------------------
#         # nfourier basis
#         # ------------------------        
#         grid_prior = self.fourier_basis(x_grid)
#         #print(grid_prior.size())
#         n_h1 = n_h1 + grid_prior
        

        # -------------------
        # cross features 
        # -------------------
        cc_n_h1 = []
        for j in range(1,self.in_channels):
            n_h1 += get_convolved_h(n_h1,nchannel=self.in_channels,p=j)

        y_out = torch.cat((h0, n_h1), dim=-1)    #(nbatch, ngrid,2*noutchannel)         
        y_out = y_out.view(-1,self.tmp_channel)  #(nbatch, ngrid,2*noutchannel) 
        y_out = self.g(y_out)
        y_out = y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)

        cc_y_out = 0
        return (y_out,cc_y_out),n_h1,h1,h0
    
    
    
    

class FinalLayer_multi(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self,  in_channels=1, out_channels = 1, nbasis = 1, init_length_scale = 1.0,min_init_length_scale=1e-6):
        super(FinalLayer_multi, self).__init__()
        
        #self.out_channels = in_channels             
        #self.in_channels_hidden = in_channels*self.out_channels       
        self.sigma_fn = torch.exp        
        self.nbasis = nbasis
        self.in_channels = in_channels        
        self.out_channels = out_channels               
                
        self.g = self.build_weight_model()

        
        self.sigma = nn.Parameter(np.log(min_init_length_scale+init_length_scale*torch.ones(self.nbasis,self.in_channels)), requires_grad=True)             #self.mu = nn.Parameter(np.log(1* torch.rand(self.nbasis,self.in_channels)), requires_grad=True)
        
        
    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """

        model = nn.Sequential(nn.Linear(self.nbasis, self.out_channels))        
        init_sequential_weights(model)
        return model
    

    
    def compute_rbf(self,x1,x2=None):
        if x2 is None:
            x2 = x1            
        # Compute shapes.            
        nbatch,npoints,nchannel = x1.size()
        
        #compute rbf over multiple channels
        dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)
        dists = dists.unsqueeze(dim=-2).repeat(1,1,1,self.nbasis,1)        
        scales = self.sigma_fn(self.sigma)[None, None, None, :,:]  
        
        dists /= (scales + eps)
        wt = torch.exp(-0.5*dists**2)   
        return wt

    
    
#     def compute_sm(self,x1,x2 = None):
#         if x2 is None:
#             x2 = x1            
#         # Compute shapes.            
#         nbatch,npoints,nchannel = x1.size()
        
#         #compute rbf over multiple channels
#         dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)
#         dists = dists.unsqueeze(dim=-2).repeat(1,1,1,self.nbasis,1)        
        
#         mu_ = self.sigma_fn(self.mu)[None, None, None,  :,:]  
#         std_ = self.sigma_fn(self.sigma)[None, None, None,  :,:]  
        
#         #exp_term_dist = dists/std_
#         exp_term_dist = dists*std_                
#         cos_term_dist = dists*mu_
        
#         wt = torch.exp(-2*(pi**2)*(exp_term_dist**2))*torch.cos(2*pi*cos_term_dist) 
#         return wt

        
        
#     #nbasis == 5 case    
#     def forward(self, x_grid, h_grid, target_x):
#         """Forward pass through the layer with evaluations at locations t.

#         Args:
#             x (tensor): Inputs of observations of shape (n, 1).
#             y (tensor): Outputs of observations of shape (n, in_channels).
#             t (tensor): Inputs to evaluate function at of shape (m, 1).

#         Returns:
#             tensor: Outputs of evaluated function at z of shape
#                 (m, out_channels).
#         """

#         nb,ntarget,nchannel = target_x.size()        
#         _,ngrid,nbasis,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
        
        
#         wt = self.compute_rbf(x_grid,target_x)
#         #wt = self.compute_sm(x_grid,target_x)
        
        
#         h = h_grid[:,:,None,:] #(nb,ngrid,1,nbasis,nchannels)
#         h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
        
#         h_out = h_out.transpose(-2,-1) #(nb,ntarget,nchannels,nbasis)
#         h_out = self.g(h_out).squeeze() #(nb,ntarget,nchannels,1)
        
        
#         if h_out.dim() == 2:
#             h_out = h_out.unsqueeze(dim=0)            
#         return h_out
    

    
    #nbasis == 5 case    
    def forward(self, x_grid, h_grid, target_x,option = 'mean'):
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
        _,ngrid,nbasis,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
        
        
        wt = self.compute_rbf(x_grid,target_x)
        #wt = self.compute_sm(x_grid,target_x)
        
        if option == 'mean':        
            h = h_grid[:,:,None,:] #(nb,ngrid,1,nbasis,nchannels)
            h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
            
        if option == 'std':
            h = h_grid[:,:,None,:]*h_grid[:,None,:,:] #(nb,ngrid,ngrid,nbasis,nchannels)            
            #print(h.size(),wt.size())
            h_out = torch.einsum('bggxy,bgtxy -> bgtxy',h,wt) #(nb,ngrid,ntarget,nbasis,nchannels)   
            #print(h_out.size())
            h_out = (h_out**2).sum(dim=1)  #(nb,ntarget,nbasis,nchannels)  

        
        
        h_out = h_out.transpose(-2,-1) #(nb,ntarget,nchannels,nbasis)
        h_out = self.g(h_out).squeeze() #(nb,ntarget,nchannels,1)
        #h_out = h_out.sum(dim=-1) #(nb,ntarget,nchannels,nbasis) #not good
        
        
        if h_out.dim() == 2:
            h_out = h_out.unsqueeze(dim=0)            
        return h_out
    
    
    
    
            


    
if __name__ == "__main__":
    model_m = ConvCNP_Multi(rho = UNet(),points_per_unit=128)

    
    

# ----------------------------------------------
# ----------------------------------------------
    
    
    
# # ----------------------------------
# # baseline model
# # ----------------------------------


# class ConvCNP_Multi(nn.Module):
#     """One-dimensional ConvCNP model.

#     Args:
#         rho (function): CNN that implements the translation-equivariant map rho.
#         points_per_unit (int): Number of points per unit interval on input.
#             Used to discretize function.
#     """

#     def __init__(self, in_channels, rho, points_per_unit,rbf_init_l=None, nbasis=5):
#         super(ConvCNP_Multi, self).__init__()
#         self.activation = nn.Sigmoid()
#         self.sigma_fn = nn.Softplus()
#         self.rho = rho
#         self.multiplier = 2 ** self.rho.num_halving_layers

#         # Compute initialisation.
#         self.points_per_unit = points_per_unit
#         if rbf_init_l is None:
#             init_length_scale = 2.0 / self.points_per_unit            
#         else:
#             init_length_scale = rbf_init_l
        
#         # Instantiate encoder
#         self.in_channels = in_channels
# #         self.encoder = ConvDeepSet_multi(in_channels = self.in_channels,
# #                                          out_channels=self.rho.in_channels,
# #                                          init_length_scale=init_length_scale)

#         # -----------
#         # consider multiple encoding features
#         # -----------
        
#         self.encoder = ConvDeepSet_multi(in_channels = self.rho.in_channels,
#                                          out_channels = self.rho.in_channels,
#                                          init_length_scale=init_length_scale)

    
# #         # Instantiate mean and standard deviation layers
# #         self.mean_layer = FinalLayer_multi(in_channels=self.rho.out_channels,
# #                                            init_length_scale=init_length_scale)
# #         self.logstd_layer = FinalLayer_multi(in_channels=self.rho.out_channels,
# #                                             init_length_scale=init_length_scale)
        
#         #self.nbasis = 5
#         self.nbasis = nbasis 
#         linear = nn.Sequential(nn.Linear(self.rho.out_channels,self.in_channels*self.nbasis))         
#         self.linear = init_sequential_weights(linear)
#         linear_cov = nn.Sequential(nn.Linear(self.rho.out_channels,self.in_channels*self.nbasis))         
#         self.linear_cov = linear_cov
#         #self.linear_covparam = nn.Parameter(torch.randn(self.rho.out_channels,self.in_channels*self.nbasis)) 
        
        
#         #nn.Sequential(nn.Linear(self.nbasis, self.out_channels)) 
# #         self.mean_layer = FinalLayer_multi(in_channels=self.in_channels,
# #                                            nbasis = self.nbasis,
# #                                            init_length_scale=init_length_scale)
# #         self.logstd_layer = FinalLayer_multi(in_channels=self.in_channels,
# #                                              nbasis = self.nbasis,
# #                                              init_length_scale=init_length_scale)
#         min_init_length_scale = 0.0
#         #init_length_scale = 0.
    
#         self.sigma = nn.Parameter(np.log(min_init_length_scale+init_length_scale*torch.rand(self.nbasis,self.in_channels)), requires_grad=True)     
#         self.mean_layer = FinalLayer_multi(self.sigma,
#                                            in_channels=self.in_channels,
#                                            nbasis = self.nbasis,
#                                            init_length_scale=init_length_scale,
#                                            option = 'mean')
#         self.logstd_layer = FinalLayer_multi(self.sigma,
#                                              in_channels=self.in_channels,
#                                              nbasis = self.nbasis,
#                                              init_length_scale=init_length_scale,
#                                              option = 'var')


#     def compute_xgrid(self,x,y,x_out,x_thres=0.1):
#         #x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - 1.0
#         #x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + 1.0
#         x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - x_thres
#         x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + x_thres
        
#         num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))
#         #print('num_points : {} in baseline'.format(num_points))
        
        
        
#         if x.is_cuda:
#             x_grid = torch.linspace(x_min, x_max, num_points).to(device)
#         else:
#             x_grid = torch.linspace(x_min, x_max, num_points)
            
#         # context
#         nb,npoints,nchannel = x.size()        
#         x_grid = x_grid[None, :, None].repeat(nb, 1, nchannel)
#         return x_grid
        
        
        

# #     def compute_hgrid(self,h):
# #         h = self.activation(h) 
# #         h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
# #         nb,_,ngrid = h.size()    
        
        
# #         h = self.rho(h)
# #         h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
# #         nb,ngrid,_= h_grid.size()
# #         h_grid = h_grid.reshape(-1,h_grid.size(-1))
# #         h_grid = self.linear(h_grid).reshape(nb,ngrid,self.nbasis,-1)        
        
# #         return h_grid
                

#     def compute_hgrid_tmp(self,h,option = 'mean',init_var=1.,eps=1e-6):
#         if option == 'mean':
#             h = self.activation(h) 
#             h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
#             nb,_,ngrid = h.size()            

#             h = self.rho(h)
#             h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
#             nb,ngrid,_= h_grid.size()
#             h_grid = h_grid.reshape(-1,h_grid.size(-1))
#             h_grid = self.linear(h_grid).reshape(nb,ngrid,self.nbasis,-1)        
#             return h_grid
        
#         if option == 'var':
#             nb,ngrid,_= h.size()
#             #h_grid = init_var*torch.log(1+ F.softplus(-h))            
#             #h_grid = init_var*torch.log(1+ torch.exp(-h))
#             #h_grid = self.linear_cov(h_grid).reshape(nb,ngrid,self.nbasis,-1)       
            
# #             h_grid = init_var*(1-torch.sigmoid(h+eps))
# #             linear_cov_pos =  F.softplus(self.linear_covparam)
# #             h_grid = F.softplus(h_grid.matmul(linear_cov_pos))
# #             h_grid = h_grid.reshape(nb,ngrid,self.nbasis,-1)                               
# #             return h_grid

#             h_grid = init_var*(1-torch.tanh(h+eps)).reshape(nb,ngrid,-1,self.in_channels)  
#             h_grid = h_grid.mean(dim=-2,keepdim=True)
#             #h_grid = init_var*(1-torch.tanh(h+eps)).reshsape()
    
#             #h_grid = F.softplus(self.linear_cov(h_grid))
#             #print(h_grid.size())
        
#             h_grid = h_grid.repeat(1,1,self.nbasis,1)        
#             #h_grid = h_grid.reshape(nb,ngrid,self.nbasis,-1)                               
#             return h_grid

        
#     def forward(self, x, y, x_out):
#         """Run the model forward.

#         Args:
#             x (tensor): Observation locations of shape (batch, data, features).
#             y (tensor): Observation values of shape (batch, data, outputs).
#             x_out (tensor): Locations of outputs of shape (batch, data, features).
            
#         Returns:
#             tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
#         """
#         nb,npoints,nchannel = x.size()        
#         x_grid = self.compute_xgrid(x,y,x_out)
#         n_h1,h1,h0 = self.encoder(x,y,x_grid)
        
        
#         # original 
#         #compute hgrid
#         #h_grid = self.compute_hgrid(n_h1)
#         #mean = self.mean_layer(x_grid,h_grid,x_out)
#         #std = self.sigma_fn(self.logstd_layer(x_grid,h_grid,x_out))   
#         #return mean, std

#         # ------------
#         # cross features 
#         # ------------
#         #cc_n_h1 = get_convolved_h(n_h1,nchannel=self.in_channels,p=1)
        
#         cc_n_h1_p1 = get_convolved_h(n_h1,nchannel=self.in_channels,p=1)
#         cc_n_h1_p2 = get_convolved_h(n_h1,nchannel=self.in_channels,p=2)
        
#         #n_h1 = n_h1 + cc_n_h1
#         #n_h1 = n_h1 + ((1-h0).clamp(min=1e-8,max=None))*0.5*(cc_n_h1_p1+cc_n_h1_p1)

#         #n_h1 = cc_n_h1
        
#         #n_h1 += n_h1
        
        
#         h_grid_mean = self.compute_hgrid_tmp(n_h1,option='mean')        
#         h_grid_var = self.compute_hgrid_tmp(h0,option='var')                
#         mean = self.mean_layer(x_grid,h_grid_mean,x_out)
#         std = self.sigma_fn(self.logstd_layer(x_grid,h_grid_var,x_out))   
#         return mean, std
    
    
    

#     def forward_encoder(self,x, y, x_out):
#         x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - 1.0
#         x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + 1.0
#         num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))
        
#         if x.is_cuda:
#             x_grid = torch.linspace(x_min, x_max, num_points).to(device)
#         else:
#             x_grid = torch.linspace(x_min, x_max, num_points)
            
#         # context
#         nbatch,npoints,nchannel = x.size()        
#         x_grid = x_grid[None, :, None].repeat(nbatch, 1, nchannel)

        

#         h = self.encoder(x,y,x_grid)
#         h = self.activation(h) 
#         h = h.permute(0, 2, 1)
#         h = self.rho(h)
#         h_grid = h.reshape(h.shape[0], -1, x_grid.size(1)).permute(0, 2, 1)
        
#         return h_grid,x_grid
    
    
    
#     @property
#     def num_params(self):
#         """Number of parameters in model."""
#         return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])




# ###########################################    
# # encoder and decoder
# ###########################################

# import math
# pi = math.pi

# eps = 1e-6
# class ConvDeepSet_multi(nn.Module):
#     """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

#     Args:
#         out_channels (int): Number of output channels.
#         init_length_scale (float): Initial value for the length scale.
#     """

#     def __init__(self, in_channels,out_channels, init_length_scale,min_init_length_scale=0.001):
#         super(ConvDeepSet_multi, self).__init__()
#         self.in_channels = in_channels        
#         self.out_channels = out_channels
#         self.g = self.build_weight_model()
        
        
#         #self.sigma = nn.Parameter(np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True)
#         #self.sigma = nn.Parameter(np.log(init_length_scale) * torch.rand(self.in_channels), requires_grad=True)
#         #self.mu = nn.Parameter(np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True)
        
#         #self.sigma = nn.Parameter(np.log(init_length_scale* torch.rand(self.in_channels)), requires_grad=True)  
#         #self.sigma = nn.Parameter(np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True)
#         self.sigma = nn.Parameter(torch.log(min_init_length_scale+init_length_scale*torch.rand(self.in_channels)), requires_grad=True)        
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

    
    
    
    
# #     def compute_sm(self,x1,x2 = None):
# #         if x2 is None:
# #             x2 = x1            
# #         # Compute shapes.            
# #         nbatch,npoints,nchannel = x1.size()
        
# #         #compute rbf over multiple channels
# #         dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)        
# #         mu_ = self.sigma_fn(self.mu)[None, None, None, :]  
# #         std_ = self.sigma_fn(self.sigma)[None, None, None, :]  
        
# #         #exp_term_dist = dists/std_
# #         exp_term_dist = dists*std_        
# #         cos_term_dist = dists*mu_
        
# #         wt = torch.exp(-2*(pi**2)*(exp_term_dist**2))*torch.cos(2*pi*cos_term_dist) 
# #         return wt

        
        
        
        
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
# #         # Compute shapes.
# #         nbatch,npoints,nchannel = context_x.size()
        
# #         #compute rbf over multiple channels
# #         dists = context_x.unsqueeze(dim=2) - x_grid.unsqueeze(dim=1)
# #         scales = self.sigma_fn(self.sigma)[None, None, None, :]        
# #         dists /= (scales + eps)


#         nbatch,npoints,nchannel = context_x.size()
#         wt,factors = self.compute_rbf(context_x,x_grid)
#         #wt = self.compute_sm(context_x,x_grid)

#         h0 = wt.sum(dim=1)

#         #print('context_y.size(),wt.size()')        
#         #print(context_y.size(),wt.size())
#         if factors > 1:
#             context_y = context_y.repeat(1,1,factors)
        
        
        
#         h1 = (context_y.unsqueeze(dim=-2)*wt).sum(dim=1)
#         n_h1 = h1/(h0+eps)        
        
# #         y_out = torch.cat((h0, n_h), dim=-1)    #(nbatch, ngrid,2*noutchannel)         
# #         y_out = y_out.view(-1,2*self.in_channels)  #(nbatch, ngrid,2*noutchannel) 
# #         y_out = self.g(y_out)
# #         y_out = y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)

#         #print('y_out.size() {}'.format(y_out.size()))
#         return n_h1,h1,h0
    

    
    

# class FinalLayer_multi(nn.Module):
#     """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

#     Args:
#         in_channels (int): Number of inputs channels.
#         init_length_scale (float): Initial value for the length scale.
#     """

#     def __init__(self, sigma,  in_channels=1, out_channels = 1, nbasis = 1, init_length_scale = 1.0,min_init_length_scale=0.001,option = 'mean'):
#         super(FinalLayer_multi, self).__init__()
        
#         #self.out_channels = in_channels             
#         #self.in_channels_hidden = in_channels*self.out_channels       
#         self.sigma_fn = torch.exp        
#         self.nbasis = nbasis
#         self.in_channels = in_channels        
#         self.out_channels = out_channels               
                
#         self.g = self.build_weight_model()

# #         self.sigma = nn.Parameter(torch.log(init_length_scale*torch.ones(self.nbasis,self.in_channels)), requires_grad=True)            
# #         self.mu = nn.Parameter(np.log(1) * torch.rand(self.nbasis,self.in_channels), requires_grad=True)
        

        
#         #self.sigma = nn.Parameter(np.log(min_init_length_scale+init_length_scale*torch.rand(self.nbasis,self.in_channels)), requires_grad=True)           
#         self.sigma = sigma
# #         self.sigma = nn.Parameter(np.log(min_init_length_scale+init_length_scale*torch.ones(self.nbasis,self.in_channels)), requires_grad=True)         
        
#         self.mu = nn.Parameter(np.log(1* torch.rand(self.nbasis,self.in_channels)), requires_grad=True)
#         self.option = option
        
#     def build_weight_model(self):
#         """Returns a function point-wise function that transforms the
#         (in_channels + 1)-dimensional representation to dimensionality
#         out_channels.

#         Returns:
#             torch.nn.Module: Linear layer applied point-wise to channels.
#         """
# #         model = nn.Sequential(
# #             nn.Linear(self.in_channels, self.out_channels),
# #         )


#         model = nn.Sequential(nn.Linear(self.nbasis, self.out_channels))        
#         init_sequential_weights(model)
#         return model
    
# #     def rbf(self, dists):
# #         """Compute the RBF values for the distances using the correct length
# #         scales.

# #         Args:
# #             dists (tensor): Pair-wise distances between x and t.

# #         Returns:
# #             tensor: Evaluation of psi(x, t) with psi an RBF kernel.
# #         """
# #         # Compute the RBF kernel, broadcasting appropriately.
# #         scales = self.sigma_fn(self.sigma)[None, None, None, :]
# #         a, b, c = dists.shape
# #         return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)


    
#     def compute_rbf(self,x1,x2=None):
#         if x2 is None:
#             x2 = x1            
#         # Compute shapes.            
#         nbatch,npoints,nchannel = x1.size()
        
#         #compute rbf over multiple channels
#         dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)
#         dists = dists.unsqueeze(dim=-2).repeat(1,1,1,self.nbasis,1)        
#         scales = self.sigma_fn(self.sigma)[None, None, None, :,:]  
        
#         dists /= (scales + eps)
#         wt = torch.exp(-0.5*dists**2)   
#         return wt

    
    
#     def compute_sm(self,x1,x2 = None):
#         if x2 is None:
#             x2 = x1            
#         # Compute shapes.            
#         nbatch,npoints,nchannel = x1.size()
        
#         #compute rbf over multiple channels
#         dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)
#         dists = dists.unsqueeze(dim=-2).repeat(1,1,1,self.nbasis,1)        
        
#         mu_ = self.sigma_fn(self.mu)[None, None, None,  :,:]  
#         std_ = self.sigma_fn(self.sigma)[None, None, None,  :,:]  
        
#         #exp_term_dist = dists/std_
#         exp_term_dist = dists*std_                
#         cos_term_dist = dists*mu_
        
#         wt = torch.exp(-2*(pi**2)*(exp_term_dist**2))*torch.cos(2*pi*cos_term_dist) 
#         return wt

        
        
#     #nbasis == 5 case    
#     def forward(self, x_grid, h_grid, target_x):
#         """Forward pass through the layer with evaluations at locations t.

#         Args:
#             x (tensor): Inputs of observations of shape (n, 1).
#             y (tensor): Outputs of observations of shape (n, in_channels).
#             t (tensor): Inputs to evaluate function at of shape (m, 1).

#         Returns:
#             tensor: Outputs of evaluated function at z of shape
#                 (m, out_channels).
#         """

#         nb,ntarget,nchannel = target_x.size()        
#         _,ngrid,nbasis,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
        
        
#         wt = self.compute_rbf(x_grid,target_x)
#         #wt = self.compute_sm(x_grid,target_x)
        
        
#         h = h_grid[:,:,None,:] #(nb,ngrid,1,nbasis,nchannels)
#         h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
        
#         h_out = h_out.transpose(-2,-1) #(nb,ntarget,nchannels,nbasis)
        
#         if self.option == 'mean':
#             h_out = self.g(h_out).squeeze() #(nb,ntarget,nchannels,1)
#         if self.option == 'var':
#             h_out = h_out.sum(dim=-1)
        
#         if h_out.dim() == 2:
#             h_out = h_out.unsqueeze(dim=0)            
#         return h_out
    
    
    
    
    


    
# if __name__ == "__main__":
#     model_m = ConvCNP_Multi(rho = UNet(),points_per_unit=128)
