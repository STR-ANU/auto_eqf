import numpy as np
from pylie import LieGroup
from abc import ABC, abstractmethod


def numericalDifferential(f, x) -> np.ndarray:
    if isinstance(x, float):
        x = np.reshape([x], (1, 1))
    h = 1e-6
    fx = f(x)
    n = fx.shape[0]
    m = x.shape[0]
    Df = np.zeros((n, m))
    for j in range(m):
        ej = np.zeros((m, 1))
        ej[j, 0] = 1.0
        Df[:, j:j+1] = (f(x + h * ej) - f(x - h * ej)) / (2*h)
    return Df


class CoordinateChart(ABC):
    @abstractmethod
    def apply(self, xi, xi0) -> np.ndarray:
        pass

    @abstractmethod
    def apply_inv(self, eps: np.ndarray, xi0):
        pass


class AutomaticEqF:
    def __init__(self, xi0, Sigma0: np.ndarray, chart: CoordinateChart, LieGroupType: type, phi, X_hat_0=None):
        assert issubclass(
            chart, CoordinateChart), "The coordinate chart provided must be derived from the abstract base class CoordinateChart."
        assert issubclass(
            LieGroupType, LieGroup), "The Lie group provided must be derived from the abstract base class LieGroup."
        eps0 = chart.apply(xi0, xi0)
        np.testing.assert_allclose(eps0, np.zeros_like(
            eps0), atol=1e-5, err_msg="The provided coordinate chart is not zero at xi0")
        assert len(
            eps0.shape) == 2 and eps0.shape[1] == 1, "The coordinate chart must return vectors."
        assert (np.linalg.eigvals(Sigma0) > 0).all(
        ), "The initial eigenvalues of Sigma must be positive."
        np.testing.assert_allclose(
            Sigma0, Sigma0.T, err_msg="Sigma must be a symmetric matrix.", atol=1e-6)

        self._xi0 = xi0
        self._Sigma = Sigma0.copy()
        if X_hat_0 is None:
            self._X_hat = LieGroupType.identity()
        else:
            self._X_hat = X_hat_0

        assert isinstance(phi(self._X_hat, self._xi0), type(
            self._xi0)), "The provided phi must return values of the same type as xi0."
        self._LieGroupType = LieGroupType
        self._epsilon = chart
        self._phi = phi
        self._vector_shape = eps0.shape
        self._liealg_shape = self._X_hat.log().shape

        # Obtain a suitable correction lift
        def phi_exp(u): return self._epsilon.apply(
            self._phi(self._LieGroupType.exp(u), self._xi0), self._xi0)
        self._g2m = numericalDifferential(
            phi_exp, np.zeros(self._liealg_shape))
        self._m2g = self._g2m.T @ np.linalg.inv(self._g2m @ self._g2m.T)

    def state_estimate(self):
        return self._phi(self._X_hat, self._xi0)

    def errorCoords(self, eta):
        return self.RHat * eta

    def propagate(self, vel: np.ndarray, vel_cov: np.ndarray, dt: float, lift):
        A0t = self.stateMatrixA(lift, vel)
        Bt = self.inputMatrixB(lift, vel)

        Q = Bt @ vel_cov @ Bt.T
        A0t_int = np.identity(self._vector_shape[0]) + dt * A0t

        # Propagation
        self._X_hat = self._X_hat * \
            self._LieGroupType.exp(lift(self.state_estimate(), vel) * dt)
        self._Sigma = A0t_int @ self._Sigma @ A0t_int.T + Q * dt

    def update(self, meas: np.ndarray, meas_cov: np.ndarray, output_map, output_action=None):
        y_hat = output_map(self.state_estimate())
        if output_action is None:
            Ct = self.outputMatrixC_standard(output_map)
        else:
            Ct = self.outputMatrixC_equivariant(meas, y_hat, output_action)
        R = meas_cov

        # Update
        yTilde = meas - y_hat
        SInv = np.linalg.inv(Ct @ self._Sigma @ Ct.T + R)
        Delta = self._m2g @ self._Sigma @ Ct.T @ SInv @ yTilde

        self._X_hat = self._LieGroupType.exp(Delta) * self._X_hat
        self._Sigma -= self._Sigma @ Ct.T @ SInv @ Ct @ self._Sigma

    def stateMatrixA(self, lift, vel: np.ndarray) -> np.ndarray:
        xi_hat = self.state_estimate()

        def epsilon_dot(eps: np.ndarray) -> np.ndarray:
            e = self._epsilon.apply_inv(eps, self._xi0)
            xi = self._phi(self._X_hat, e)
            lift_v_tilde = lift(xi, vel) - lift(xi_hat, vel)
            xi_1 = self._phi(self._LieGroupType.exp(lift_v_tilde), xi_hat)
            e_1 = self._phi(self._X_hat.inv(), xi_1)
            eps_1 = self._epsilon.apply(e_1, self._xi0)
            return eps_1

        A0t = numericalDifferential(epsilon_dot, np.zeros(self._vector_shape))
        return A0t

    def inputMatrixB(self, lift, vel: np.ndarray) -> np.ndarray:
        xi_hat = self.state_estimate()

        def epsilon_dot(mu: np.ndarray) -> np.ndarray:
            lift_v_tilde = lift(xi_hat, vel + mu) - lift(xi_hat, vel)
            xi_hat_1 = self._phi(self._LieGroupType.exp(lift_v_tilde), xi_hat)
            e_1 = self._phi(self._X_hat.inv(), xi_hat_1)
            eps_1 = self._epsilon.apply(e_1, self._xi0)
            return eps_1

        Bt = numericalDifferential(epsilon_dot, np.zeros_like(vel))
        return Bt

    def outputMatrixC_standard(self, h) -> np.ndarray:
        def y_fun(eps):
            e = self._epsilon.apply_inv(eps, self._xi0)
            xi = self._phi(self._X_hat, e)
            y = h(xi)
            return y

        Ct = numericalDifferential(y_fun, np.zeros(self._vector_shape))
        return Ct

    def outputMatrixC_equivariant(self, y_hat, y, rho) -> np.ndarray:
        # Assume the innovation lift provides the identification of coordinates
        def rho_avg(u):
            r_y = rho(self._LieGroupType.exp(u), y)
            r_y_hat = rho(self._LieGroupType.exp(u), y_hat)
            return 0.5 * (r_y + r_y_hat)

        Drho = numericalDifferential(rho_avg, np.zeros(self._liealg_shape))
        Ct = Drho @ self._X_hat.inv().Adjoint() @ self._m2g
        return Ct

    def filter_energy(self, xi) -> float:
        e = self._phi(self._X_hat.inv(), xi)
        eps = self._epsilon.apply(e, self._xi0)
        err = eps.T @ np.linalg.inv(self._Sigma) @ eps
        return float(err)
