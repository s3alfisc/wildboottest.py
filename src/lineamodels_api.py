from functools import cached_property
from linearmodels.panel.covariance import HomoskedasticCovariance
from linearmodels.typing import Float64Array, IntArray
from typing import Union
from boot_algo3 import boot_algo3

class WildBootstrap(HomoskedasticCovariance):
    
    def __init__(
        self,
        y: Float64Array,
        x: Float64Array,
        params: Float64Array,
        entity_ids: Union[IntArray, None],
        time_ids: Union[IntArray, None],
        *,
        debiased: bool = False,
        extra_df: int = 0,
    ):
        self._y = y
        self._x = x
        self._params = params
        self._entity_ids = entity_ids
        self._time_ids = time_ids
        self._debiased = debiased
        self._extra_df = extra_df
        self._nobs, self._nvar = x.shape
        self._nobs_eff = self._nobs - extra_df
        if debiased:
            self._nobs_eff -= self._nvar
        self._scale = self._nobs / self._nobs_eff
        self._name = "Wild Bootstrap"

    @property
    def name(self) -> str:
        """Covariance estimator name"""
        return self._name

    @property
    def eps(self) -> Float64Array:
        """Model residuals"""
        return self._y - self._x @ self._params

    @property
    def s2(self) -> float:
        """Error variance"""
        eps = self.eps
        return self._scale * float(eps.T @ eps) / self._nobs

    @cached_property
    def cov(self) -> Float64Array:
        # add in bootalgo code

    def deferred_cov(self) -> Float64Array:
        """Covariance calculation deferred until executed"""
        return self.cov