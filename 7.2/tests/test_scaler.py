import unittest
import numpy as np
from scaler import MyScaler

class TestScaler(unittest.TestCase):
    np.random.seed(1984)
    X = np.random.random((100,1))
    X_new = np.random.random((100,1))
    X2 = np.random.random((100, 2))
    X2 = np.random.random((100,2))
    scaler = MyScaler()
    
    def test_scaler(self):
        self.scaler.fit(self.X, None)
        transformed = self.scaler.transform(self.X)
        self.assertAlmostEqual(transformed.mean(), 0)
        self.assertAlmostEqual(np.std(transformed), 1)
    
    def test_transform(self):
        self.scaler.fit(self.X, None)
        #Check behavior on new data.
        new_transform = self.scaler.transform(self.X_new)
        self.assertAlmostEqual(np.mean(new_transform), 0.2473700079)
        self.assertAlmostEqual(np.std(new_transform), 0.890837722840)
        
    def test_multiple_columns(self):
        self.scaler.fit(self.X2, None)
        transformed = self.scaler.transform(self.X2)
        map(lambda x: self.assertAlmostEqual(*x), 
            zip(np.mean(transformed, 0), np.array([0,0])))
        
        map(lambda x: self.assertAlmostEqual(*x),
           zip(np.std(transformed, 0), np.array([1,1])))