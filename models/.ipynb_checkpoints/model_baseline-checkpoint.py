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






    
    
    
# ----------------------------------
# baseline model
# ----------------------------------
class Convcnp_multioutput(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    #def __init__(self, in_channels, rho, points_per_unit,rbf_init_l=None, nbasis=5):
    def __init__(self, in_channels, points_per_unit=64,rbf_init_l=0.1, nbasis=5):
        
        super(Convcnp_multioutput, self).__init__()
        self.modelname = 'baseline'
        
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        #self.rho = rho
        #self.multiplier = 2 ** self.rho.num_halving_layers
        self.multiplier = 2 ** 3

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        if rbf_init_l is None:
            init_length_scale = 2.0 / self.points_per_unit            
        else:
            init_length_scale = rbf_init_l
        
        # Instantiate encoder
        self.in_channels = in_channels
#         self.encoder = ConvDeepSet_multioutput(in_channels = self.in_channels,
#                                               out_channels=self.rho.in_channels,
#                                               init_length_scale=init_length_scale)

        self.encoder = ConvDeepSet(in_channels = self.in_channels,
                                   out_channels = 8,
                                   init_length_scale=init_length_scale)
          
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 16, 5, 1, 2),            
            nn.ReLU(),            
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),            
            nn.Conv1d(16, 8, 5, 1, 2),
        )                
        
        def weights_init(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
        self.cnn.apply(weights_init)
                
        
        
    

        #self.nbasis = 5
        self.nbasis = nbasis 
        #linear = nn.Sequential(nn.Linear(self.rho.out_channels,self.in_channels*self.nbasis))
        linear = nn.Sequential(nn.Linear(8,self.in_channels*self.nbasis))
        self.linear = init_sequential_weights(linear)        
        self.mean_layer = FinalLayer_multioutput(in_channels=self.in_channels,
                                                nbasis = self.nbasis,
                                                init_length_scale=init_length_scale)

        self.logstd_layer = FinalLayer_multioutput(in_channels=self.in_channels,
                                                   nbasis = self.nbasis,
                                                   init_length_scale=init_length_scale)



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
        
        
        

    def compute_hgrid(self,h):
        h = self.activation(h) 
        h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
        nb,_,ngrid = h.size()    
        #print('h.size()')
        #print(h.size())
        h = self.cnn(h)
        h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
        nb,ngrid,_= h_grid.size()
        h_grid = h_grid.reshape(-1,h_grid.size(-1))
        h_grid = self.linear(h_grid).reshape(nb,ngrid,self.nbasis,-1)        
        
        return h_grid
                

        
    def forward(self, x, y, x_out):
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
        concat_n_h1h0,n_h1,h1,h0 = self.encoder(x,y,x_grid)        
                
        h_grid = self.compute_hgrid(concat_n_h1h0)
        mean = self.mean_layer(x_grid,h_grid,x_out)
        std = self.sigma_fn(self.logstd_layer(x_grid,h_grid,x_out))   
        return mean, std 
    
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])






class Convcnp_multioutput2(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    #def __init__(self, in_channels, rho, points_per_unit,rbf_init_l=None, nbasis=5):
    def __init__(self, in_channels, points_per_unit=64,rbf_init_l=0.1, nbasis=3):
        
        super(Convcnp_multioutput2, self).__init__()
        self.modelname = 'baseline'
        
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        #self.rho = rho
        #self.multiplier = 2 ** self.rho.num_halving_layers
        self.multiplier = 2 ** 3

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        if rbf_init_l is None:
            init_length_scale = 2.0 / self.points_per_unit            
        else:
            init_length_scale = rbf_init_l
        
        # Instantiate encoder
        self.in_channels = in_channels
#         self.encoder = ConvDeepSet_multioutput(in_channels = self.in_channels,
#                                               out_channels=self.rho.in_channels,
#                                               init_length_scale=init_length_scale)

        self.encoder = ConvDeepSet(in_channels = self.in_channels,
                                   out_channels = 8,
                                   init_length_scale=init_length_scale)
          
        self.cnn = nn.Sequential(
            nn.Conv1d(8, 16, 5, 1, 2),            
            nn.ReLU(),            
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),            
            nn.Conv1d(16, 8, 5, 1, 2),
        )                
        
        def weights_init(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
        self.cnn.apply(weights_init)
                
        
        
    

        #self.nbasis = 5
        self.nbasis = nbasis 
        #linear = nn.Sequential(nn.Linear(self.rho.out_channels,self.in_channels*self.nbasis))
        #linear = nn.Sequential(nn.Linear(8,self.in_channels*self.nbasis))
        linear = nn.Sequential(nn.Linear(8,2*self.in_channels*self.nbasis))
        
        self.linear = init_sequential_weights(linear)        
        self.mean_layer = FinalLayer_multioutput(in_channels=self.in_channels,
                                                nbasis = self.nbasis,
                                                init_length_scale=init_length_scale)

        self.logstd_layer = FinalLayer_multioutput(in_channels=self.in_channels,
                                                   nbasis = self.nbasis,
                                                   init_length_scale=init_length_scale)



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
        
        
        

    def compute_hgrid(self,h):
        h = self.activation(h) 
        h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
        nb,_,ngrid = h.size()    
        #print('h.size()')
        #print(h.size())
        h = self.cnn(h)
        h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
        nb,ngrid,_= h_grid.size()
        h_grid = h_grid.reshape(-1,h_grid.size(-1))
        #h_grid = self.linear(h_grid).reshape(nb,ngrid,2*self.nbasis,self.in_channels)        
        h_grid = self.linear(h_grid).reshape(nb,ngrid,-1,self.in_channels)        
        
        return h_grid
                

        
    def forward(self, x, y, x_out):
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
        concat_n_h1h0,n_h1,h1,h0 = self.encoder(x,y,x_grid)        
                
        hgrid = self.compute_hgrid(concat_n_h1h0)
        hgrid_mean,hgrid_std = hgrid.split((self.nbasis,self.nbasis),dim=-2)
        mean = self.mean_layer(x_grid,hgrid_mean,x_out)
        std = self.sigma_fn(self.logstd_layer(x_grid,hgrid_std,x_out))   
        return mean, std
    
    
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])

    
    

###########################################    
# encoder and decoder
###########################################

import math
pi = math.pi

eps = 1e-6
class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels,out_channels, init_length_scale=1.0,min_init_length_scale=1e-6):
        super(ConvDeepSet, self).__init__()
        self.in_channels = in_channels        
        self.out_channels = out_channels
        self.g = self.build_weight_model()               
        self.sigma = nn.Parameter(torch.log(min_init_length_scale+init_length_scale*torch.ones(self.in_channels)), requires_grad=True)        
        #self.mu = nn.Parameter(np.log(1* torch.rand(self.in_channels)), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        #model = nn.Sequential(nn.Linear(self.in_channels, self.out_channels))
        model = nn.Sequential(nn.Linear(2*self.in_channels, self.out_channels))
        
        init_sequential_weights(model)
        return model
    
        
    
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

        
        
        
    def forward(self, context_x, context_y, x_grid):
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
        
        y_out = torch.cat((h0, n_h1), dim=-1)    #(nbatch, ngrid,2*noutchannel)         
        y_out = y_out.view(-1,2*self.in_channels)  #(nbatch, ngrid,2*noutchannel) 
        y_out = self.g(y_out)
        y_out = y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)

        #return y_out,h1,h0
        return y_out,n_h1,h1,h0
    

    
    

class FinalLayer_multioutput(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self,  in_channels=1, out_channels = 1, nbasis = 1, init_length_scale = 1.0,min_init_length_scale=1e-6):
        super(FinalLayer_multioutput, self).__init__()
        
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
        _,ngrid,nbasis,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
        
        
        wt = self.compute_rbf(x_grid,target_x)
        #wt = self.compute_sm(x_grid,target_x)
        
        
        h = h_grid[:,:,None,:] #(nb,ngrid,1,nbasis,nchannels)
        h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
        
        h_out = h_out.transpose(-2,-1) #(nb,ntarget,nchannels,nbasis)
        h_out = self.g(h_out).squeeze() #(nb,ntarget,nchannels,1)
        
        
        if h_out.dim() == 2:
            h_out = h_out.unsqueeze(dim=0)            
        return h_out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
###########################################    
# configuration for parameters
#mu_scale = 0.1
#init_length_scale = 0.01 #sin3
#init_length_scale = 0.5 #matern 3



# from torch.fft import rfftn, irfftn,rfft, irfft

# def get_convolved_h(h_grid,nchannel=3,p=None):    
#     nb,ngrid,nkernels = h_grid.size()
#     #permuted_index = np.random.permutation(nkernels)        
    
#     #p = np.random.randint(1,nkernels)
#     #p=6    
#     if p is None:
#         p = np.random.randint(1,nchannel)
#     permuted_index = np.roll(np.arange(nkernels),-p)
    
#     #print(p,np.arange(nkernels),permuted_index)
#     return convolve_1d_functional_h(h_grid,h_grid[:,:,permuted_index])



# def convolve_1d_functional_h(h_i,h_j,target_dim=1,eps=1e-4):
#     """
#     inputs
#         h_i : (nb,ngrids,nchannel)
#         h_j : (nb,ngrids,nchannel)    
#     outputs
#         h_ij : (nb,ngrids,nchannel)
#     """
    
    
#     if h_i.size(-1) > h_j.size(-1):
#         p2d = (0,h_i.size(-1) - h_j.size(-1))
#         h_j = F.pad(h_j, p2d, "constant", 0)
    
    
    
#     #nhiddens = h_i.size(1)
#     nhiddens = h_i.size(target_dim)
    
#     if nhiddens % 2 != 0:
#         nhiddens -= 1
#         h_i = h_i[:,:-1,:]
#         h_j = h_j[:,:-1,:]    
    
#     f_h_i = rfft(h_i,dim=target_dim)
#     f_h_j = rfft(h_j,dim=target_dim)
    

#     #plt.scatter(i,(torch.roll(a,shifts=i,dims=-1)*b).sum(dim=1))
#     #f_h_i.imag *= -1    
#     #n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim)        
    
#     #plt.scatter(i,(a*torch.roll(b,shifts=i,dims=-1)).sum(dim=1))
#     f_h_j.imag *= -1    
#     n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim)        
#     #n_ij_conv = irfft((f_h_i*f_h_j),dim=target_dim) / (h_i.norm(dim=-1,keepdim=True)*h_j.norm(dim=-1,keepdim=True) + eps)       
    
#     #return n_ij_conv/(n_ij_conv.norm(dim=-1,keepdim=True) + eps)
#     #return n_ij_conv/(h_i.norm(dim=-1,keepdim=True)*h_j.norm(dim=-1,keepdim=True) + eps)
#     return (n_ij_conv-n_ij_conv.mean(dim=1,keepdim=True))/(n_ij_conv.std(dim=1,keepdim=True) + eps)


    
    
    


    
if __name__ == "__main__":
    model_m = ConvCNP_Multi(rho = UNet(),points_per_unit=128)
