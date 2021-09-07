from pylie import SO3
from automatic_eqf import *

import matplotlib.pyplot as plt

e1 = np.array([[1.0], [0.0], [0.0]])


class SphereNormalCoordinates(CoordinateChart):
    @staticmethod
    def apply(xi, xi0=None) -> np.ndarray:
        if xi0 is not None:
            np.testing.assert_allclose(xi0, e1)
        # sphere coordinates around e3 based on SO3.
        sin_th = np.linalg.norm(SO3.skew(e1) @ xi)
        cos_th = e1.T @ xi

        th = np.arctan2(sin_th, cos_th)
        if abs(th) < 1e-8:
            omega = SO3.skew(e1) @ xi * 1.0
        else:
            omega = SO3.skew(e1) @ xi * (th / sin_th)

        omega = omega[1:, :]
        return omega

    @staticmethod
    def apply_inv(eps: np.ndarray, xi0=None):
        if xi0 is not None:
            np.testing.assert_allclose(xi0, e1)
        omega = np.vstack((0.0, eps))
        xi = SO3.exp(-omega) @ e1
        return xi


def phi(R: SO3, eta):
    return R.inv() * eta


def lift(eta, omega):
    return omega


if __name__ == '__main__':
    # Set noise parameters for simulation
    np.random.seed(0)
    max_t = 5.0
    init_noise = 0.5
    gyr_noise = 0.01
    meas_noise = 0.05
    dt = 0.01
    apply_gm_noise = True

    # Initialise the state and filter
    eta = e1 + init_noise * np.random.randn(3, 1)
    eta = eta / np.linalg.norm(eta)

    Sigma0 = np.eye(2) * init_noise**2
    sphere_eqf = AutomaticEqF(e1, Sigma0, SphereNormalCoordinates, SO3, phi)

    # Prepare to record data
    tru_states = []
    est_states = []
    filter_energy = []

    max_step = int(max_t / dt)
    for step in range(max_step):
        ct = step * dt

        # Update the state
        omega = np.array([
            [0.1 * np.cos(2*ct)],
            [0.2 * np.sin(ct)],
            [0.0]
        ])
        eta = SO3.exp(-dt * omega) @ eta

        # Measure noisy values
        gyr = omega + np.random.randn(3, 1) * gyr_noise * apply_gm_noise
        meas = eta + np.random.randn(3, 1) * meas_noise * apply_gm_noise

        # Apply the filter
        sphere_eqf.propagate(gyr, np.eye(3) * gyr_noise**2, dt, lift)
        sphere_eqf.update(meas, np.eye(3) * meas_noise**2, lambda x: x, phi)

        # Record data
        tru_states.append(eta.copy())
        est_states.append(sphere_eqf.state_estimate())
        filter_energy.append(sphere_eqf.filter_energy(eta))

    # Plot the data collected
    times = tuple(s*dt for s in range(max_step))
    angle_error = tuple(float(180.0 / np.pi * abs(np.arccos(y1.T @ y2)))
                        for (y1, y2) in zip(tru_states, est_states))
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(times, angle_error, 'r-')
    ax[1].plot(times, filter_energy, 'r-')

    ax[0].set_ylabel("Angle Error (deg)")
    ax[0].set_xticks([])
    ax[1].set_ylabel("Lyapunov Value")
    ax[1].set_xlabel("Time (s)")

    for a in ax:
        a.set_xlim(times[0], times[-1])
        a.set_yscale('log')

    fig.suptitle("Automatic EqF for single-bearing estimation")
    fig.tight_layout()

    plt.show()
