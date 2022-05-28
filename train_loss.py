import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal as MVN



def sum_from_nth_dim(t, dim):
    """Sum all dims from `dim`. E.g. sum_after_nth_dim(torch.rand(2,3,4,5), 2).shape = [2,3]"""
    return t.view(*t.shape[:dim], -1).sum(-1)


def sum_log_prob(prob, sample):
    """Compute log probability then sum all but the z_samples and batch."""    
    log_p = prob.log_prob(sample)          # size = [n_z_samples, batch_size, *]    
    sum_log_p = sum_from_nth_dim(log_p, 2) # size = [n_z_samples, batch_size]
    return sum_log_p

    

#def compute_nll_latent_loss( p_yCc, z_samples, q_zCc, q_zCct, Y_trgt):
#def compute_nll_latent_loss( pred_mu,pred_std, Y_trgt):
def compute_nll_latent( pred_mu,pred_std, target_y, z_samples=None, qz_c=None, qz_ct=None):
    
    """
    compute NLLLossLNPF
    # computes approximate LL in a numerically stable way
    # LL = E_{q(z|y_cntxt)}[ \prod_t p(y^t|z)]
    # LL MC = log ( mean_z ( \prod_t p(y^t|z)) )
    # = log [ sum_z ( \prod_t p(y^t|z)) ] - log(n_z_samples)
    # = log [ sum_z ( exp \sum_t log p(y^t|z)) ] - log(n_z_samples)
    # = log_sum_exp_z ( \sum_t log p(y^t|z)) - log(n_z_samples)
    
    """
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

    #print(sum_log_w_k.size())
    # log_sum_exp_z ... . size = [batch_size]
    log_S_z_sum_p_yCz = torch.logsumexp(sum_log_w_k, 0)
    # - log(n_z_samples)
    log_E_z_sum_p_yCz = log_S_z_sum_p_yCz - math.log(n_z_samples)    

    # NEGATIVE log likelihood
    #return -log_E_z_sum_p_yCz
    return -log_E_z_sum_p_yCz.mean()  #averages each loss over batches 





#def gaussian_logpdf(inputs, mean, sigma, reduction=None):
def compute_nll( mean, sigma ,targets_y, reduction='batched_mean'):
    
    """Gaussian log-density.

    Args:
        inputs (tensor): Inputs.
        mean (tensor): Mean.
        sigma (tensor): Standard deviation.
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    dist = Normal(loc=mean, scale=sigma)
    logp = dist.log_prob(targets_y)
    #print(logp.size())

    if not reduction:
        return logp
    elif reduction == 'sum':
        return -torch.sum(logp)
    elif reduction == 'mean':
        return -torch.mean(logp)
    elif reduction == 'batched_mean':
        #return -torch.mean(torch.sum(logp, 1))
        return -torch.mean(torch.sum(logp, dim=(1,2)))
    
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

        
        

def mvn_logpdf(mean , cov , target_y, reduction=None):
    """multivariate_Gaussian log-density.

    Args:
        inputs (tensor): (nbatch,nobs)
        mean (tensor): (nbatch,nobs)
        sigma (tensor): (nbatch,nobs,nobs)
        reduction (str, optional): Reduction. Defaults to no reduction.
            Possible values are "sum", "mean", and "batched_mean".

    Returns:
        tensor: Log-density.
    """
    #dist = Normal(loc=mean, scale=sigma)
    #logp = dist.log_prob(inputs)
    y_mean = torch.zeros_like(target_y)
    dist = MVN(loc = y_mean , covariance_matrix=cov)
    logp = dist.log_prob(inputs)
    
    if not reduction:
        return logp
    elif reduction == 'sum':
        return torch.sum(logp)
    elif reduction == 'mean':
        return torch.mean(logp)
    elif reduction == 'batched_mean':
        return torch.mean(torch.sum(logp, 1))
    else:
        raise RuntimeError(f'Unknown reduction "{reduction}".')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        