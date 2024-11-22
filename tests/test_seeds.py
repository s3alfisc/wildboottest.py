import pytest
import pandas as pd
import statsmodels.formula.api as sm
from wildboottest.wildboottest import wildboottest
import numpy as np

@pytest.fixture
def data():
  np.random.seed(12312)
  N = 1000
  k = 3
  G = 20
  # small sample size -> full enumeration
  X = np.random.normal(0, 1, N * k).reshape((N,k))
  X[:,0] = 1
  beta = np.random.normal(0,1,k)
  beta[1] = 0.005
  u = np.random.normal(0,1,N)
  Y = X @ beta + u
  cluster = np.random.choice(list(range(0,G)), N)
  X_df = pd.DataFrame(X)
  Y_df = pd.DataFrame(Y)
  cluster_df = pd.DataFrame(cluster)
  df = pd.concat([X_df, Y_df, cluster_df], axis = 1)  
  df.columns = ['intercept','X1','X2','Y', 'cluster']

  return df

def test_results_from_same_seed(data):
    
    model = sm.ols(formula='Y ~ X1 + X2', data=data)    

    cluster_list = [data.cluster, None]
    for x in cluster_list: 

        # same seed used in function -> same results
        a = wildboottest(model, param = "X1", cluster = x, B= 999, seed=876587)
        b = wildboottest(model, param = "X1", cluster = x, B= 999, seed=876587)
        pd.testing.assert_frame_equal(a,b)
        
        # random seed outside of function 2x -> same results
        np.random.seed(123)
        a2 = wildboottest(model, param = "X1", cluster = x, B= 999)
        np.random.seed(123)
        b2 = wildboottest(model, param = "X1", cluster = x, B= 999)
        pd.testing.assert_frame_equal(a2,b2)

def test_seeds_and_rng(data):
    model = sm.ols(formula='Y ~ X1 + X2', data=data)    

    cluster_list = [data.cluster, None]
    
    for x in cluster_list: 

        # specifying seed and rng with that seed -> same results
        a = wildboottest(model, param = "X1", cluster = x, B= 999, seed=876587)
        rng = np.random.default_rng(seed=876587)
        b = wildboottest(model, param = "X1", cluster = x, B= 999, seed=rng)
        pd.testing.assert_frame_equal(a,b)