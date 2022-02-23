import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
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





class Convcnplatent_multiouput(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, in_channels, rho, points_per_unit,rbf_init_l=None, nbasis=5,num_samples=10):
        super(Convcnplatent_multiouput, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        if rbf_init_l is None:
            init_length_scale = 2.0 / self.points_per_unit            
        else:
            init_length_scale = rbf_init_l
        
        # Instantiate encoder
        self.in_channels = in_channels
        self.encoder = ConvDeepSet_multioutput(in_channels = self.in_channels,
                                               out_channels=self.rho.in_channels,
                                               init_length_scale=init_length_scale)
    
        self.num_samples = num_samples
        self.nbasis = nbasis 
        linear = nn.Sequential(nn.Linear(self.rho.out_channels,2*self.in_channels*self.nbasis))         
        self.linear = init_sequential_weights(linear)
        self.interpolator =  FinalLayer_multioutput(in_channels=self.in_channels,
                                                    nbasis = self.nbasis,
                                                    init_length_scale=init_length_scale)


        
        linear_out = nn.Sequential(nn.Linear(self.in_channels*self.nbasis,self.in_channels*2))         
        self.linear_out = init_sequential_weights(linear_out)
        
        
        

    def compute_xgrid(self,x,y,x_out,x_thres=0.1):
        x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - x_thres
        x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + x_thres        
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(x.device)
        
        # context
        nb,npoints,nchannel = x.size()        
        x_grid = x_grid[None, :, None].repeat(nb, 1, nchannel)
        return x_grid
        
        
        

    def compute_hgrid(self,h):
        h = self.activation(h) 
        h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
        nb,_,ngrid = h.size()    
                
        h = self.rho(h)
        h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
        nb,ngrid,_= h_grid.size()
        h_grid = h_grid.reshape(-1,h_grid.size(-1))
        h_grid = self.linear(h_grid).reshape(nb,ngrid,-1)        
        #h_grid = self.linear(h_grid)        
        
        return h_grid
                
        
    def samples_z(self,h_mu,h_std,num_samples=10):        
        h_mu = h_mu[None,:,:,:]
        h_std = 0.1+0.9*torch.sigmoid(h_std)[None,:,:,:]        
        eps = torch.randn(num_samples,h_std.size(1),h_std.size(-1))
        eps = Variable(eps[:,:,None,:]).to(h_mu.device)
        
        z_samples =  h_mu + h_std*eps        
        return z_samples,h_mu.squeeze(),h_std.squeeze()

        
        
        
        
    def forward(self, x, y, x_out, y_out=None):
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
        assert h_grid.size(-1) % 2 == 0
        h_grid_mu,h_grid_std = h_grid.split(h_grid.size(-1)//2,dim=-1)
        z_samples_c,h_mu_c,h_std_c = self.samples_z(h_grid_mu,h_grid_std,num_samples=self.num_samples)
        z_samples = collapse_z_samples_batch(z_samples_c)
        z_samples = z_samples.reshape(z_samples.size(0),z_samples.size(1),self.nbasis,-1)
        
        if y_out is not None:
            x_grid_t = self.compute_xgrid(x_out,y_out,x_out)            
            concat_n_h1h0_t,_,_,_ = self.encoder(x_out,y_out,x_grid_t)        
            h_grid_t = self.compute_hgrid(concat_n_h1h0_t)
            assert h_grid_t.size(-1) % 2 == 0
            h_grid_mu_t,h_grid_std_t = h_grid_t.split(h_grid_t.size(-1)//2,dim=-1)
            z_samples_t,h_mu_t,h_std_t = self.samples_z(h_grid_mu_t,h_grid_std_t,num_samples=self.num_samples)
            
                
        
        x_grid = collapse_z_samples_batch(replicate_z_samples(x_grid, n_z_samples=self.num_samples))        
        x_out = collapse_z_samples_batch(replicate_z_samples(x_out, n_z_samples=self.num_samples))         
        h_out = self.interpolator(x_grid,z_samples,x_out)
        h_out = h_out.reshape(h_out.size(0),h_out.size(1),-1)
        
        # ----------------
        # linear decoder
        # ----------------        
        h_out = self.linear_out(h_out)        
        y_mu,y_logstd = h_out.split(self.in_channels,dim=-1)
        
        
        if y_mu.dim() == 3:
            y_mu = y_mu.reshape(self.num_samples,nb,y_mu.size(-2),y_mu.size(-1))
            y_logstd = y_logstd.reshape(self.num_samples,nb,y_logstd.size(-2),y_logstd.size(-1))
        
        
        if y_out is not None:
            return y_mu, 0.01+0.99*F.softplus(y_logstd),z_samples_t,(h_mu_c,h_std_c),(h_mu_t,h_std_t)

        if y_out is None:
            return y_mu, 0.01+0.99*F.softplus(y_logstd)
    
    
    
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
class ConvDeepSet_multioutput(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels,out_channels, init_length_scale=1.0,min_init_length_scale=1e-6):
        super(ConvDeepSet_multioutput, self).__init__()
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

        
#        self.sigma = nn.Parameter(np.log(min_init_length_scale+init_length_scale*torch.ones(self.nbasis*self.in_channels)), requires_grad=True)             #self.mu = nn.Parameter(np.log(1* torch.rand(self.nbasis,self.in_channels)), requires_grad=True)
        
        
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
        _,ngrid,_,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
        
        wt = self.compute_rbf(x_grid,target_x)        
        h = h_grid[:,:,None,:,:] #(nb,ngrid,1,nbasis,nchannels)
        h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
        return h_out
    

    
if __name__ == "__main__":
    model_m = ConvCNP_Multi(rho = UNet(),points_per_unit=128)
