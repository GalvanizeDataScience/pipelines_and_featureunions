from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class MyScaler(TransformerMixin, BaseEstimator):
    """Scale to zero mean and unit variance.
    """
    def fit(self, X, y):
        """Recommended signature for custom transformer's
        fit method.
        
        Set state in your transformer with whatever information
        is needed to transform later.
        """
        #You have to return self, so we can chain!
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        """Recommended signature for custom transformer's
        transform method.
        
        Use state (if any) to transform some X data. This X
        may be the same X passed to fit, but it may also be new data,
        as in the case of a CV dataset. Both are treated the same.
        """
        #Do transforms.
        Xt = X.copy()
        Xt -= self.mean
        Xt /= self.scale
        return Xt