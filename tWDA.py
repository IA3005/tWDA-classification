from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.extmath import softmax

from pyriemann.utils.base import logm,_matrix_operator

from joblib import Parallel, delayed

import numpy as np  


from tWishartEstimation import RCG

    
    
class tWDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by t-Wishart.
    """
    
    def __init__(self,n,df,n_jobs=1):
        """Init."""
        self.n = n #nb of time samples
        self.df = df
        self.n_jobs = n_jobs
        
    
    def compute_class_center(self,S,df):
        _,p,_ = S.shape
        if df==np.inf:
            return np.mean(S,axis=0)/self.n
        return RCG(S,self.n,df=df)


    def fit(self, S, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : tWDA classifier instance
        """
        self.classes_ = np.unique(y)
        Nc = len(self.classes_)
        
        y = np.asarray(y)
        p,_ = S[0].shape
        if self.n_jobs==1:
            self.centers = [self.compute_class_center(S[y==self.classes_[i]],self.df) for i in range(Nc)]
        else:
            self.centers = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_class_center)(S[y==self.classes_[i]],self.df) for i in range(Nc))
        self.pi = np.ones(Nc)
        
        for k in range(Nc):
            self.pi[k]= len(y[y==self.classes_[k]])/len(y)
        
        return self
 
    
    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.centers)
        K,p,_ =covtest.shape
        dist = np.zeros((K,Nc)) #shape= (n_trials,n_classes)
        
        for i in range(Nc):
            if (self.df==np.inf):
                log_h = lambda t:-0.5*t
            else:
                log_h = lambda t:-0.5*(self.df+self.n*p)*np.log(1+t/self.df)
                    #if you use different df per class, add to log_h this ter
                    ##-0.5*self.n*p*np.log(0.5*self.dfs[i])-betaln(self.dfs[i]/2,self.n*p/2)+gammaln(self.n*p/2)
               
            center = self.centers[i].copy()
            inv_center = _matrix_operator(center,lambda x : 1/x)
            logdet_center = np.trace(logm(center))
            for j in range(K):
                #distance between the center of the class i and the cov_j
                dist[j,i] = np.log(self.pi[i])-0.5*self.n*logdet_center+log_h(np.matrix.trace(inv_center@covtest[j]))
        return dist

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        covtest : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        preds = []
        n_trials,n_classes = dist.shape
        for i in range(n_trials):
            preds.append(self.classes_[dist[i,:].argmax()])
        preds = np.asarray(preds)
        return preds

    def transform(self, S):
        """get the distance to each centroid.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(S)

    def fit_predict(self, S, y):
        """Fit and predict in one function."""
        self.fit(S, y)
        return self.predict(S)

    def predict_proba(self, S):
        """Predict proba using softmax.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self._predict_distances(S)**2)




