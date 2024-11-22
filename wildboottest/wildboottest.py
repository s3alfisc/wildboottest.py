from __future__ import annotations # add so that we can use type annotations as strings to get rid of circular imports
import numpy as np
import pandas as pd
from numba import jit, njit, prange
from wildboottest.weights import draw_weights
import warnings
from typing import Union, Tuple, Callable
from numpy.random import Generator
from statsmodels.regression.linear_model import OLS


_allowed_models = (
  OLS,
)

class WildDrawFunctionException(Exception):
    pass

class TestMatrixNonConformabilityException(Exception):
  pass

class TestBootstrapTypeException(Exception):
  pass

class TestHCImposeNullException(Exception):
  pass

class TestHCWeightsException(Exception):
  pass

class WildboottestHC:

    """Create an object of WildboottestHC and get p-value by successively applying
    methods in the following way:

    Example:

      >>> import numpy as np
      >>> from wildboottest.wildboottest import WildboottestHC
      >>> np.random.seed(12312312)
      >>> N = 1000
      >>> k = 3
      >>> G = 10
      >>> X = np.random.normal(0, 1, N * k).reshape((N,k))
      >>> beta = np.random.normal(0,1,k)
      >>> beta[0] = 0.005
      >>> u = np.random.normal(0,1,N)
      >>> Y = 1 + X @ beta + u
      >>> R = np.array([1, 0, 0])
      >>> r = 0
      >>> B = 999
      >>> wb = WildboottestHC(X = X, Y = Y, R = R, r = 0, B = B)
      >>> wb.get_adjustments(bootstrap_type = '11')
      >>> wb.get_uhat(impose_null = True)
      >>> wb.get_tboot(weights_type = "rademacher")
      >>> wb.get_tstat()
      >>> wb.get_pvalue()
    """

    def __init__(self, X : Union[np.ndarray, pd.DataFrame, pd.Series],
          Y: Union[np.ndarray, pd.DataFrame, pd.Series],
          R : Union[np.ndarray, pd.DataFrame],
          r: Union[np.ndarray, float],
          B: int,
          seed:  Union[int, Generator, None] = None) -> None:

        """Initializes the Heteroskedastic Wild Bootstrap Class

        Args:
          X (Union[np.ndarray, pd.DataFrame, pd.Series]): Exogeneous variable array or dataframe
          Y (Union[np.ndarray, pd.DataFrame, pd.Series]): Endogenous variable array or dataframe
          R (Union[np.ndarray, pd.DataFrame]): Constraint matrix for running bootstrap
          B (int): bootstrap iterations
          seed (Union[int, Generator, None], optional): Random seed for random weight types. 
          If an integer, will be used as a seed in a numpy default random generator, or a numpy random generator 
          can also be specified and used. Defaults to None.

        Raises:
          TypeError: Raise if input arrays are lists
          TestMatrixNonConformabilityException: Raise if constraint matrix shape does not conform to X
        """

        for i in [X, Y]:
          if isinstance(i, list):
            raise TypeError(f"{i} cannot be a list")

        if isinstance(X, (pd.DataFrame, pd.Series)):
          self.X = X.values
        else:
          self.X = X

        if isinstance(Y, (pd.DataFrame, pd.Series)):
          self.Y = Y.values
        else:
          self.Y = Y

        if isinstance(seed, int):
          self.rng = np.random.default_rng(seed=seed)
        elif isinstance(seed, Generator):
          self.rng = seed
        else:
          self.rng = np.random.default_rng()

        self.N = X.shape[0]
        self.k = X.shape[1]
        self.B = B
        self.R = R
        self.r = r

        if self.X.shape[1] != self.R.shape[0]:
          raise TestMatrixNonConformabilityException("The number of rows in the test matrix R, does not ")

    def get_adjustments(self, bootstrap_type):

        '''
        Raises:
          TestBootstrapTypeException: If non-appropriate bootstrap types are selected
        '''
        if bootstrap_type not in ['11', '21', '31']:
            raise TestBootstrapTypeException("For the heteroskedastic (i.e. non-clustered) wild bootstrap, only types '11', '21' and '31' are supported.")


        self.tXXinv = np.linalg.inv(np.transpose(self.X) @ self.X)
        self.resid_multiplier_boot, self.small_sample_correction = _adjust_scores(self.X, self.tXXinv, bootstrap_type[0])

    def get_uhat(self, impose_null : bool):

        '''
        Raises:
          TestHCImposeNullException: If the null is not imposed on the bootstrap dgp
        '''
        if impose_null is not True:
          raise TestHCImposeNullException('For the heteroskedastic bootstrap, the null needs to be imposed.')


        self.tXy = np.transpose(self.X) @ self.Y
        self.beta_hat = self.tXXinv @ self.tXy
        self.uhat = self.Y - self.X @ self.beta_hat

        if impose_null:
          self.impose_null = True
          self.beta_r = self.beta_hat - self.tXXinv @ self.R * ( 1 / (np.transpose(self.R) @ self.tXXinv @ self.R)) * (np.transpose(self.R) @ self.beta_hat - self.r)#self.uhat_r = self.Y - self.beta_r
          self.uhat_r = self.Y - self.X @ self.beta_r
          self.uhat2 = self.uhat_r * self.resid_multiplier_boot
        else:
          self.impose_null = False
          self.uhat2 = self.uhat * self.resid_multiplier_boot

    def get_tboot(self, weights_type: Union[str, Callable]):

        if weights_type not in ['rademacher', 'norm']:
            raise TestHCWeightsException("For the heteroskedastic bootstrap, only weight tyes 'rademacher' and 'normal' are supported, but you provided '" + weights_type + "' .")
        self.weights_type = weights_type

        self.tXXinvX = self.tXXinv @ np.transpose(self.X)

        if self.impose_null == True:
          beta = self.beta_r.reshape((self.k, 1))
        else:
          beta = self.beta_hat.reshape((self.k, 1))

        yhat = (self.X @ beta).flatten()

        R = self.R.reshape((self.k, 1)).astype("float")
        self.RXXinvX_2 = np.power(np.transpose(R) @ self.tXXinv @ np.transpose(self.X), 2)
        #RXXinv_2 = np.power(np.transpose(R) @ self.tXXinv, 2)

        #@jit
        def _run_hc_bootstrap(B, weights_type, N, X, yhat, uhat2, tXXinv, RXXinvX_2, Rt, small_sample_correction, rng):

            t_boot = np.zeros(B)
            tXXinvX = tXXinv @ np.transpose(X)

            for b in range(B):
            # create weights vector. mammen weights not supported via numba
                v, _ = draw_weights(
                          t = weights_type,
                          full_enumeration = False,
                          N_G_bootcluster = N,
                          boot_iter = 1,
                          rng=rng
                        )

                v = v.flatten()

                uhat_boot = uhat2 * v
                yhat_boot = yhat + uhat_boot
                beta_boot = tXXinvX  @ yhat_boot
                resid_boot = yhat_boot - X @ beta_boot
                cov_v = small_sample_correction * RXXinvX_2 @ np.power(resid_boot, 2)
                t_boot[b] = (Rt @ beta_boot / np.sqrt(cov_v))[0]

            return t_boot

        self.t_boot = _run_hc_bootstrap(
            B = self.B,
            weights_type = self.weights_type,
            N = self.N,
            X = self.X,
            yhat = yhat,
            uhat2 = self.uhat2,
            tXXinv = self.tXXinv,
            RXXinvX_2 = self.RXXinvX_2,
            Rt = np.transpose(R),
            small_sample_correction=self.small_sample_correction,
            rng = self.rng
          )

    def get_tstat(self):

        cov = self.small_sample_correction * self.RXXinvX_2 @ np.power(self.uhat, 2)
        self.t_stat = (np.transpose(self.R) @ self.beta_hat - self.r) / np.sqrt(cov)

    def get_pvalue(self, pval_type = "two-tailed"):

        if pval_type == "two-tailed":
            self.pvalue = np.mean(np.abs(self.t_stat) < abs(self.t_boot))
        elif pval_type == "equal-tailed":
            pl = np.mean(self.t_stat < self.t_boot)
            ph = np.mean(self.t_stat > self.t_boot)
            self.pvalue = 2 * min(pl, ph)
        elif pval_type == ">":
            self.pvalue = np.mean(self.t_stat < self.t_boot)
        else:
            self.pvalue = np.mean(self.t_stat > self.t_boot)


def _adjust_scores(X, tXXinv, variant):

    N = X.shape[0]
    k = X.shape[1]

    if variant == "1":
      # HC1
      resid_multiplier = np.ones(N)
      small_sample_correction = (N-1) / (N-k)
    else:
      hatmat = X @ tXXinv @ np.transpose(X)
      diag_hatmat = np.diag(hatmat)
      small_sample_correction = 1
      if variant == "2":
        # HC2
        resid_multiplier = 1 / np.sqrt(1-diag_hatmat)
      elif variant == "3":
        # HC3
        resid_multiplier = 1 / (1-diag_hatmat)

    return resid_multiplier, small_sample_correction

class WildboottestCL:
  """Create an object of WildboottestCL and get p-value by successively applying
  methods in the following way:

  Example:

      >>> import numpy as np
      >>> from wildboottest.wildboottest import WildboottestCL
      >>> np.random.seed(12312312)
      >>> N = 1000
      >>> k = 3
      >>> G = 10
      >>> X = np.random.normal(0, 1, N * k).reshape((N,k))
      >>> beta = np.random.normal(0,1,k)
      >>> beta[0] = 0.005
      >>> u = np.random.normal(0,1,N)
      >>> Y = 1 + X @ beta + u
      >>> cluster = np.random.choice(list(range(G)), N)
      >>> R = np.array([1, 0, 0])
      >>> B = 999

      >>> wb = WildboottestCL(X = X, Y = Y, cluster = cluster, R = R, B = B)
      >>> wb.get_scores(bootstrap_type = "11", impose_null = True)
      >>> wb.get_weights(weights_type= "rademacher")
      >>> wb.get_numer()
      >>> wb.get_denom()
      >>> wb.get_tboot()
      >>> wb.get_vcov()
      >>> wb.get_tstat()
      >>> wb.get_pvalue()
  """

  def __init__(self, X : Union[np.ndarray, pd.DataFrame, pd.Series],
               Y: Union[np.ndarray, pd.DataFrame, pd.Series],
               cluster : Union[np.ndarray, pd.DataFrame, pd.Series],
               R : Union[np.ndarray, pd.DataFrame],
               B: int,
               bootcluster: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None,
               seed:  Union[int, Generator, None] = None,
               parallel: bool = True) -> None:
    """Initializes the Wild Cluster Bootstrap Class

    Args:
        X (Union[np.ndarray, pd.DataFrame, pd.Series]): Exogeneous variable array or dataframe
        Y (Union[np.ndarray, pd.DataFrame, pd.Series]): Endogenous variable array or dataframe
        cluster (Union[np.ndarray, pd.DataFrame, pd.Series]): Cluster array or dataframe
        R (Union[np.ndarray, pd.DataFrame]): Constraint matrix for running bootstrap
        B (int): bootstrap iterations
        bootcluster (Union[np.ndarray, pd.DataFrame, pd.Series, None], optional): Sub-cluster array. Defaults to None.
        seed (Union[int, Generator, None], optional): Random seed for random weight types. 
          If an integer, will be used as a seed in a numpy default random generator, or a numpy random generator 
          can also be specified and used. Defaults to None.        
        parallel (bool, optional): Whether to run the bootstrap in parallel. Defaults to True.
    Raises:
        TypeError: Raise if input arrays are lists
        TestMatrixNonConformabilityException: Raise if constraint matrix shape does not conform to X
    """

    "Initialize the WildboottestCL class"
    #assert bootstrap_type in ['11', '13', '31', '33']
    #assert impose_null in [True, False]

    self.parallel = parallel

    if bootcluster is None:
      bootcluster = cluster

    for i in [X, Y, cluster, bootcluster]:
      if isinstance(i, list):
        raise TypeError(f"{i} cannot be a list")

    if isinstance(X, (pd.DataFrame, pd.Series)):
      self.X = X.values
    else:
      self.X = X

    if isinstance(Y, (pd.DataFrame, pd.Series)):
      self.Y = Y.values
    else:
      self.Y = Y

    if isinstance(cluster, pd.DataFrame):
      self.clustid = cluster.unique()
      self.cluster = cluster.values
    if isinstance(bootcluster, pd.DataFrame):
      self.bootclustid = bootcluster.unique()
      self.bootcluster = bootcluster.values
    else:
      self.clustid = np.unique(cluster)
      self.bootclustid = np.unique(bootcluster)
      self.bootcluster = bootcluster

    if isinstance(seed, int):
      self.rng = np.random.default_rng(seed=seed)
    elif isinstance(seed, Generator):
      self.rng = seed
    else:
      self.rng = np.random.default_rng()
      
    self.N_G_bootcluster = len(self.bootclustid)
    self.G  = len(self.clustid)

    self.N = X.shape[0]
    self.k = X.shape[1]
    self.B = B
    self.R = R
    self.r = 0

    if self.X.shape[1] != self.R.shape[0]:
      raise TestMatrixNonConformabilityException("The number of rows in the test matrix R, does not ")

    X_list = []
    y_list = []
    tXgXg_list = []
    tXgyg_list = []
    tXX = np.zeros((self.k, self.k))
    tXy = np.zeros(self.k)

    #all_cluster = np.unique(bootcluster)

    for g in self.bootclustid:

      # split X and Y by (boot)cluster
      X_g = self.X[np.where(self.bootcluster == g)]
      Y_g = self.Y[np.where(self.bootcluster == g)]
      tXgXg = np.transpose(X_g) @ X_g
      tXgyg = np.transpose(X_g) @ Y_g
      X_list.append(X_g)
      y_list.append(Y_g)
      tXgXg_list.append(tXgXg)
      tXgyg_list.append(tXgyg)
      tXX += tXgXg
      tXy += tXgyg

    self.X_list = X_list
    self.Y_list = y_list
    self.tXgXg_list = tXgXg_list
    self.tXgyg_list = tXgyg_list
    self.tXX = tXX
    self.tXy = tXy

    self.tXXinv = np.linalg.inv(tXX)
    self.RtXXinv = np.matmul(R, self.tXXinv)

  def get_weights(self, weights_type: Union[str, Callable]) -> Tuple[np.ndarray, int, bool]:
    """Function for getting weights for bootstrapping.

    Args:
        weights_type (Tuple[str, Callable]): The distribution to be used. Accepts Either 'rademacher', 'mammen', 'norm' or 'webb'. Optionally accepts a callable of one argument, `n`, the number of bootstraps iterations.

    Returns:
        Tuple[np.ndarray, int]: Returns the arrays of weights and the number of bootstrap iterations
    """
    self.weights_type = weights_type

    if 2**self.N_G_bootcluster < self.B and weights_type=='rademacher':
      self.full_enumeration = True
      full_enumeration_warn=True
    else:
      self.full_enumeration = False
      full_enumeration_warn=False

    self.v, self.B = draw_weights(
      t = self.weights_type,
      full_enumeration = self.full_enumeration,
      N_G_bootcluster = self.N_G_bootcluster,
      boot_iter = self.B,
      rng=self.rng
    )

    return self.v, self.B, full_enumeration_warn

  def get_scores(self, bootstrap_type : str,
                 impose_null : bool, adj: bool = True,
                 cluster_adj: bool = True) -> np.ndarray:
    """Run bootstrap and get scores for each variable

    Args:
        bootstrap_type (str): Determines which wild cluster bootstrap type should be run. Options are "fnw11","11", "13", "31" and "33" for the wild cluster bootstrap and "11" and "31" for the heteroskedastic bootstrap. For more information, see the details section. "fnw11" is the default for the cluster bootstrap, which runs a "11" type wild cluster bootstrap via the algorithm outlined in "fast and wild" (Roodman et al (2019)). "11" is the default for the heteroskedastic bootstrap.
        impose_null (bool): Controls if the null hypothesis is imposed on the bootstrap dgp or not. Null imposed (WCR) by default. If False, the null is not imposed (WCU)
        adj (bool, optional): Whether to adjust for small sample. Defaults to True.
        cluster_adj (bool, optional): Whether to do a cluster-robust small sample correction. Defaults to True.

    Returns:
        np.ndarray: The output array of scores of shape kxG
    """

    if bootstrap_type[1:2] == '1':
      self.crv_type = "crv1"
      self.ssc = 1
      if adj:
        self.ssc = self.ssc * (self.N - 1) / (self.N - self.k)
      if cluster_adj:
        self.ssc = self.ssc * self.G / (self.G - 1)
    elif bootstrap_type[1:2] == '3':
      self.crv_type = "crv3"
      self.ssc = (self.G - 1) / self.G

    bootstrap_type_x = bootstrap_type[0:1] + 'x'

    if impose_null == True:
      self.bootstrap_type = "WCR" + bootstrap_type_x
    else:
      self.bootstrap_type = "WCU" + bootstrap_type_x

    # not needed for all types, but compute anyways
    self.beta_hat = self.tXXinv @ self.tXy

    # precompute required objects for computing scores & vcov's
    if self.bootstrap_type in ["WCR3x"]:

      X = self.X
      X1 = X[:,self.R == 0]
      X1_list = []
      tX1gX1g_list = []
      tX1gyg_list = []
      tXgX1g_list = []
      tX1X1 = np.zeros((self.k-1, self.k-1))
      tX1y = np.zeros(self.k-1)

      for ix, g in enumerate(self.bootclustid):
        #ix = g = 1
        X1_list.append(X1[np.where(self.bootcluster == g)])
        tX1gX1g_list.append(np.transpose(X1_list[ix]) @ X1_list[ix])
        tX1gyg_list.append(np.transpose(X1_list[ix]) @ self.Y_list[ix])
        tXgX1g_list.append(np.transpose(self.X_list[ix]) @  X1_list[ix])
        tX1X1 = tX1X1 + tX1gX1g_list[ix]
        tX1y = tX1y + tX1gyg_list[ix]

      beta_1g_tilde = []

      for ix, g in enumerate(self.bootclustid):
        beta_1g_tilde.append(np.linalg.pinv(tX1X1 - tX1gX1g_list[ix]) @ (tX1y - tX1gyg_list[ix]))

      beta = beta_1g_tilde
      M = tXgX1g_list

    elif self.bootstrap_type in ["WCU3x"]:

      beta_g_hat = []
      for ix, g in enumerate(self.bootclustid):
        beta_g_hat.append(np.linalg.pinv(self.tXX - self.tXgXg_list[ix]) @ (self.tXy - self.tXgyg_list[ix]))

      beta = beta_g_hat
      M = self.tXgXg_list

    elif self.bootstrap_type in ["WCR1x"]:

      A = 1 / (np.transpose(self.R) @ self.tXXinv @ self.R)
      beta_tilde = self.beta_hat - self.tXXinv @ self.R * A * (self.R @ self.beta_hat - 0)
      beta = beta_tilde
      M = self.tXgXg_list

    elif self.bootstrap_type in ["WCU1x"]:

      beta = self.beta_hat
      M = self.tXgXg_list

    # compute scores based on tXgyg, M, beta
    scores_list = []

    if(self.bootstrap_type in ["WCR1x", "WCU1x"]):

      for ix, g in enumerate(self.bootclustid):

        scores_list.append(self.tXgyg_list[ix] - M[ix] @ beta)

    elif(self.bootstrap_type in ["WCR3x", "WCU3x"]):

      for ix, g in enumerate(self.bootclustid):

        scores_list.append(self.tXgyg_list[ix] - M[ix] @ beta[ix])

    self.scores_mat = np.transpose(np.array(scores_list)) # k x G

    return self.scores_mat


  def get_numer(self):
      # Calculate the bootstrap numerator
      self.Cg = self.R @ self.tXXinv @ self.scores_mat
      self.numer = self.Cg @ self.v

  def get_denom(self):

      if self.crv_type == "crv1":

        H = np.zeros((self.G, self.G))

        # numba optimization possible?
        for ixg, g in enumerate(self.bootclustid):
          for ixh, h in enumerate(self.bootclustid):
            # can be improved by replacing list calls with matrices;
            H[ixg,ixh] = self.R @ self.tXXinv @ self.tXgXg_list[ixg] @ self.tXXinv @ self.scores_mat[:,ixh]

        # now compute denominator
        # numba / cython / c++ optimization possible? Porting this part from
        # R to c++ gives good speed improvements
        @njit(parallel = self.parallel)
        def compute_denom(Cg, H, bootclustid, B, G, v, ssc):

          denom = np.zeros(B)

          for b in prange(B):
            Zg = np.zeros(G)
            for ixg, g in enumerate(bootclustid):
              vH = 0
              for ixh, h in enumerate(bootclustid):
                vH += v[ixh,b] * H[ixg,ixh]
              Zg[ixg] = Cg[ixg] * v[ixg,b] - vH

            # todo: ssc
            denom[b] = ssc * np.sum(np.power(Zg,2))

          return denom

        self.denom = compute_denom(self.Cg, H, self.bootclustid, self.B, self.G, self.v, self.ssc)

      elif self.crv_type == "crv3":

        self.inv_tXX_tXgXg = []
        for ix, g in enumerate(self.bootclustid):
          self.inv_tXX_tXgXg.append(np.linalg.pinv(self.tXX - self.tXgXg_list[ix]))

        self.denom = np.zeros(self.B)

        for b in prange(self.B):

          scores_g_boot = np.zeros((self.G, self.k))
          v_ = self.v[:,b]

          for ixg, g in enumerate(self.bootclustid):

            scores_g_boot[ixg,:] = self.scores_mat[:,ixg] * v_[ixg]

          scores_boot = np.sum(scores_g_boot, axis = 0)
          delta_b_star = self.tXXinv @ scores_boot

          delta_diff = np.zeros((self.G, self.k))

          for ixg, g in enumerate(self.bootclustid):

            score_diff = scores_boot - scores_g_boot[ixg,:]
            delta_diff[ixg,:] = (

              (self.inv_tXX_tXgXg[ixg] @ score_diff - delta_b_star)**2

              )
          # se's
          self.denom[b] = self.ssc * np.sum(delta_diff, axis = 0)[np.where(self.R == 1)]


  def get_tboot(self):

      self.t_boot = self.numer / np.sqrt(self.denom)

  def get_vcov(self):

    if self.crv_type == "crv1":

      meat = np.zeros((self.k,self.k))
      for ixg, g in enumerate(self.bootclustid):
        score = np.transpose(self.X_list[ixg]) @ (self.Y_list[ixg] - self.X_list[ixg] @ self.beta_hat)
        meat += np.outer(score, score)

      self.vcov = self.tXXinv @ meat @ self.tXXinv

    elif self.crv_type == "crv3":

      # calculate leave-one out beta hat
      beta_jack = np.zeros((self.G, self.k))
      for ixg, g in enumerate(self.bootclustid):
        beta_jack[ixg,:] = (
          np.linalg.pinv(self.tXX - self.tXgXg_list[ixg]) @ (self.tXy - np.transpose(self.X_list[ixg]) @ self.Y_list[ixg])
        )

      if not hasattr(self, "beta_hat"):
        beta_hat = self.tXXinv @ self.tXy

      beta_center = self.beta_hat

      vcov3 = np.zeros((self.k, self.k))
      for ixg, g in enumerate(self.bootclustid):
        beta_centered = beta_jack[ixg,:] - beta_center
        vcov3 += np.outer(beta_centered, beta_centered)

      self.vcov =  vcov3


  def get_tstat(self):

    se = np.sqrt(self.ssc * self.R @ self.vcov @ np.transpose(self.R))
    self.t_stat = (np.transpose(self.R) @ self.beta_hat - self.r )/ se

  def get_pvalue(self, pval_type = "two-tailed"):

    if pval_type == "two-tailed":
      self.pvalue = np.mean(np.abs(self.t_stat) < abs(self.t_boot))
    elif pval_type == "equal-tailed":
      pl = np.mean(self.t_stat < self.t_boot)
      ph = np.mean(self.t_stat > self.t_boot)
      self.pvalue = 2 * min(pl, ph)
    elif pval_type == ">":
      self.pvalue = np.mean(self.t_stat < self.t_boot)
    else:
      self.pvalue = np.mean(self.t_stat > self.t_boot)


def wildboottest(model : OLS,
                 B:int,
                 cluster : Union[np.ndarray, pd.Series, pd.DataFrame, None] = None,
                 param : Union[str, None] = None,
                 weights_type: str = 'rademacher',
                 impose_null: bool = True,
                 bootstrap_type: str = '11',
                 seed: Union[int, Generator, None] = None,
                 adj: bool = True,
                 cluster_adj: bool = True,
                 parallel: bool = True,
                 show=True) -> pd.DataFrame:
  """Run a wild cluster bootstrap based on an object of class 'statsmodels.regression.linear_model.OLS'

  Args:
      model (OLS):  A statsmodels regression object
      B (int): The number of bootstrap iterations to run
      cluster (Union[None, np.ndarray, pd.Series, pd.DataFrame], optional): If None (default), a 'heteroskedastic' wild boostrap
           is run. For a wild cluster bootstrap, requires a numpy array of dimension one,a  pandas Series or DataFrame, containing the clustering variable.
      param (Union[str, None], optional): A string of length one, containing the test parameter of interest. Defaults to None.
      weights_type (str, optional): The type of bootstrap weights. Either 'rademacher', 'mammen', 'webb' or 'normal'.
                        'rademacher' by default. Defaults to 'rademacher'.
      impose_null (bool, optional): Should the null hypothesis be imposed on the bootstrap dgp, or not?
                           Defaults to True.
      bootstrap_type (str, optional):A string of length one. Allows to choose the bootstrap type
                          to be run. Either '11', '31', '13' or '33'. '11' by default. Defaults to '11'.
      seed (Union[int, Generator, None], optional): Random seed for random weight types. 
        If an integer, will be used as a seed in a numpy default random generator, or a numpy random generator 
        can also be specified and used. Defaults to None.      
      adj (bool, optional): Whether to adjust for small sample. Defaults to True.
      cluster_adj (bool, optional): Whether to do a cluster-robust small sample correction. Defaults to True.
      parallel (bool, optional): Whether to run the bootstrap in parallel. Defaults to True.
      show (bool, optional): Whether to print the results. Defaults to True.

  Raises:
      Exception: Raises if `param` is not a string

  Returns:
      pd.DataFrame: A wild cluster bootstrapped p-value(s).

  Example:

      >>> from wildboottest.wildboottest import wildboottest
      >>> import statsmodels.api as sm
      >>> import numpy as np
      >>> import pandas as pd

      >>> np.random.seed(12312312)
      >>> N = 1000
      >>> k = 10
      >>> G = 10
      >>> X = np.random.normal(0, 1, N * k).reshape((N,k))
      >>> X = pd.DataFrame(X)
      >>> X.rename(columns = {0:"X1"}, inplace = True)
      >>> beta = np.random.normal(0,1,k)
      >>> beta[0] = 0.005
      >>> u = np.random.normal(0,1,N)
      >>> Y = 1 + X @ beta + u
      >>> cluster = np.random.choice(list(range(0,G)), N)
      >>> model = sm.OLS(Y, X)
      >>> wildboottest(model, param = "X1", B = 9999)
      >>> wildboottest(model, param = "X1", cluster = cluster, B = 9999)
      >>> wildboottest(model, cluster = cluster, B = 9999)
  """
  
  if not isinstance(model, _allowed_models):
    raise NotImplementedError(f"Only allow models of type {' ,'.join([str(i) for i in _allowed_models])}")

  # does model.exog already exclude missing values?
  X = model.exog
  # interestingly, the dependent variable is called 'endogeneous'
  Y = model.endog
  # weights not yet used, only as a placeholder
  weights = model.weights

  xnames = model.data.xnames
  ynames = model.data.ynames

  pvalues = []
  tstats = []

  def generate_stats(param, cluster):

      R = np.zeros(len(xnames))
      R[xnames.index(param)] = 1
      r = 0
      # Just test for beta=0

      # is it possible to fetch the clustering variables from the pre-processed data
      # frame, e.g. with 'excluding' observations with missings etc
      # cluster = ...

      if cluster is None:

          boot = WildboottestHC(X = X, Y = Y, R = R, r = r, B = B, seed = seed)
          boot.get_adjustments(bootstrap_type = bootstrap_type)
          boot.get_uhat(impose_null = impose_null)
          boot.get_tboot(weights_type = weights_type)
          boot.get_tstat()
          boot.get_pvalue(pval_type = "two-tailed")
          full_enumeration_warn = False

      else:

          boot = WildboottestCL(X = X, Y = Y, cluster = cluster,
                              R = R, B = B, seed = seed, parallel = parallel)
          boot.get_scores(bootstrap_type = bootstrap_type, impose_null = impose_null, adj=adj, cluster_adj=cluster_adj)
          _, _, full_enumeration_warn = boot.get_weights(weights_type = weights_type)
          boot.get_numer()
          boot.get_denom()
          boot.get_tboot()
          boot.get_vcov()
          boot.get_tstat()
          boot.get_pvalue(pval_type = "two-tailed")

      pvalues.append(boot.pvalue)
      tstats.append(boot.t_stat)

      return pvalues, tstats, full_enumeration_warn

  if param is None:
    for x in xnames:
      pvalues, tstats, full_enumeration_warn = generate_stats(x, cluster=cluster)
    param = xnames
  elif isinstance(param, str):
    pvalues, tstats, full_enumeration_warn = generate_stats(param, cluster=cluster)
  else:
    raise Exception("`param` not correctly specified")

  if full_enumeration_warn:
    warnings.warn("2^G < the number of boot iterations, setting full_enumeration to True.")

  res = {
    'param': param,
    'statistic': tstats,
    'p-value': pvalues
  }

  res_df = pd.DataFrame(res).set_index('param')

  if show:
    print(res_df.to_markdown(floatfmt=".3f"))

  return res_df

if __name__ == '__main__':
    import statsmodels.api as sm
    import numpy as np

    np.random.seed(12312312)
    N = 1000
    k = 10
    G= 10
    X = np.random.normal(0, 1, N * k).reshape((N,k))
    beta = np.random.normal(0,1,k)
    beta[0] = 0.005
    u = np.random.normal(0,1,N)
    Y = 1 + X @ beta + u
    cluster = np.random.choice(list(range(G)), N)

    X_df = pd.DataFrame(data=X, columns = [f"col_{i}" for i in range(k)])

    Y_df = pd.DataFrame(data=Y, columns = ['outcome'])

    model = sm.OLS(Y_df, X_df)

    print("--- wildboottest ---")

    wildboottest(model, cluster=cluster, B=9999, weights_type='rademacher',
                 impose_null=True, bootstrap_type='11')

    print("--- NonRobust ---")
    print(model.fit().summary())

    print("--- WCB ---")
    print(model.fit(cov_type='wildclusterbootstrap',
              cov_kwds = {'cluster' : cluster,
                          'B' : 9999,
                          'weights_type' : 'rademacher',
                          'impose_null' : True,
                          'bootstrap_type' : '11',
                          'seed' : None}).summary())
