import numpy as np    
import numpy.linalg as la
from scipy.linalg.lapack import dtrtri
from scipy.stats import ortho_group, norm, uniform
from scipy.special import gammaln,betaln

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient,SteepestDescent
from pymanopt.manifolds.manifold import RiemannianSubmanifold
from scipy.linalg import sqrtm,pinv

from manifold import SPD
from functools import partial
import os, sys
    


def RCG(S,n,df):
    """
    Compute the maximum likelihood estimator (MLE) of the 
    center of a t-Wishart distribution of df degrees of freedom
    denoted as t-W(n,Center,df). The MLE is derived iteratively,
    using the Riemannian Conjugate Gradient algorithm

    Parameters
    ----------
    S : ndarray, shape (n_samples, p, p)
        samples.
    n : int
        parameter of t-Wishart distribution
    df : float
        degree of freedom used for estimation.
    manifold : class
        class of the Riemannian manifold of pxp SPD matrices, endowed with 
        the FIM of t-Wishart distribution.

    Returns
    -------
    ndarray, shape (p,p)
        MLE of the center of t-Wishart.

    """
    _,p,_ = S.shape
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    
    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R,S,n,df)
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R,S,n,df)
    #
    
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    init = np.eye(S.shape[-1])
    optimizer = ConjugateGradient(verbosity=0)
    return optimizer.run(problem, initial_point=init).point

def t_wish_cost(R,S,n,df):
    """
    Parameters
    ----------
    R : ndarray, shape (p,p)
        center of distribution.
    S : ndarray, shape (n_samples,p,p)
        samples.
    n : int
        parameter of the t-Wishart distribution.
    df : float
        degrees of freedo of the t-Wishart distribution.

    Returns
    -------
    float
        negative loglikelihood of samples S drawn from t-W(n,R,df).

    """
    p, _ = R.shape
    return ellipt_wish_cost(R,S,n,partial(t_logh,df=df,dim=n*p))

def t_wish_egrad(R,S,n,df):
    """
    Parameters
    ----------
    R : ndarray, shape (p,p)
        center of distribution.
    S : ndarray, shape (n_samples,p,p)
        samples.
    n : int
        parameter of the t-Wishart distribution.
    df : float
        degrees of freedo of the t-Wishart distribution.

    Returns
    -------
    ndarray, shape (p,p)
        Euclidean gradient of the negative log-likelihood.

    """
    p, _ = R.shape
    return ellipt_wish_egrad(R,S,n,partial(t_u,df=df,dim=n*p))


def ellipt_wish_cost(R,S,n,logh):
    k, p, _ = S.shape
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(logh(a))/n/k


def ellipt_wish_egrad(R,S,n,u):
    k, p, _ = S.shape
    # psi
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij',u(a),S)
    #
    return la.solve(R,la.solve(R.T,((R  - psi/n/k) /2).T).T)

def t_logh(t,df,dim):
    return -(df+dim)/2*np.log(1+t/df) 

def t_u(t,df,dim):
    return (df+dim)/(df+t)





