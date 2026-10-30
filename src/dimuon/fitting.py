"""Peak fitting: Gaussian signal on a linear background.

A particle resonance appears as a Gaussian bump (detector resolution
convolved with the natural Breit-Wigner lineshape; for the Z the detector
resolution ~2 GeV dominates the 2.5 GeV natural width, so a Gaussian is an
excellent first model) on top of a smooth combinatorial background that is
locally linear.

The fit is a binned maximum-likelihood-style least squares with Poisson
errors sqrt(N + 1) per bin, and parameter uncertainties from the covariance
matrix of scipy.optimize.curve_fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def gaussian_plus_linear(x, amp, mu, sigma, slope, intercept):
    """Gaussian peak + linear background."""
    return (
        amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        + slope * (x - np.mean(x))
        + intercept
    )


def crystalball_plus_exp(x, n_sig, mu, sigma, beta, power, n_bkg, tau):
    """Crystal Ball signal + exponential background.

    The Crystal Ball function is a Gaussian core with a power-law tail on
    the low side -- the standard lineshape for resonances decaying to
    electrons/muons, because final-state radiation (bremsstrahlung) shifts
    reconstructed masses downward. beta is where the tail starts (in units
    of sigma) and power controls how fast it falls.
    """
    from scipy.stats import crystalball

    return (
        n_sig * crystalball.pdf(x, beta, power, loc=mu, scale=sigma)
        + n_bkg * np.exp(-(x - np.min(x)) / tau)
    )


@dataclass
class PeakFit:
    """Result of fitting one resonance peak."""

    name: str
    window: tuple[float, float]
    mass: float
    mass_err: float
    width: float          # Gaussian sigma (detector resolution dominated)
    width_err: float
    amplitude: float
    chi2_ndf: float
    n_events: int         # events in the fit window
    bin_width: float
    beta: float | None = None    # Crystal Ball tail start (sigma units)
    power: float | None = None   # Crystal Ball tail power
    signal_yield: float | None = None  # absolute signal counts (CB fits)

    @property
    def n_signal(self) -> float:
        """Signal yield in counts."""
        if self.signal_yield is not None:
            return self.signal_yield
        return self.amplitude * self.width * np.sqrt(2.0 * np.pi) / self.bin_width


def fit_peak(
    masses: np.ndarray,
    window: tuple[float, float],
    name: str = "peak",
    bins: int = 80,
    initial_mass: float | None = None,
) -> PeakFit:
    """Histogram ``masses`` in ``window`` and fit Gaussian + linear background.

    ``initial_mass`` is only a starting guess for the optimizer; the fitted
    mass is determined by the data. If None, the bin with the most counts
    is used.
    """
    counts, edges = np.histogram(masses, bins=bins, range=window)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bin_width = edges[1] - edges[0]
    err = np.sqrt(counts + 1.0)

    if initial_mass is None:
        initial_mass = centers[np.argmax(counts)]

    p0 = [counts.max(), initial_mass, (window[1] - window[0]) / 20.0, 0.0, counts.min()]
    bounds = (
        [0.0, window[0], bin_width, -np.inf, 0.0],
        [np.inf, window[1], (window[1] - window[0]) / 4.0, np.inf, np.inf],
    )
    popt, pcov = curve_fit(
        gaussian_plus_linear, centers, counts, p0=p0, sigma=err,
        absolute_sigma=True, bounds=bounds, maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    residuals = (counts - gaussian_plus_linear(centers, *popt)) / err
    ndf = len(counts) - len(popt)

    return PeakFit(
        name=name,
        window=window,
        mass=float(popt[1]),
        mass_err=float(perr[1]),
        width=float(popt[2]),
        width_err=float(perr[2]),
        amplitude=float(popt[0]),
        chi2_ndf=float(np.sum(residuals**2) / ndf),
        n_events=int(counts.sum()),
        bin_width=float(bin_width),
    )


def fit_peak_asymmetric(
    masses: np.ndarray,
    window: tuple[float, float],
    name: str = "peak",
    bins: int = 80,
    initial_mass: float | None = None,
) -> PeakFit:
    """Fit a Crystal Ball signal + exponential background.

    Use this when the peak has a physical low-mass tail (Z -> mu+ mu- with
    bremsstrahlung). Parameter bounds keep the optimizer in physical space.
    """
    from scipy.stats import crystalball

    counts, edges = np.histogram(masses, bins=bins, range=window)
    centers = 0.5 * (edges[1:] + edges[:-1])
    bin_width = edges[1] - edges[0]
    err = np.sqrt(counts + 1.0)
    w_lo, w_hi = window
    span = w_hi - w_lo

    if initial_mass is None:
        initial_mass = centers[np.argmax(counts)]

    total = counts.sum()
    p0 = [0.6 * total, initial_mass, span / 20.0, 1.5, 3.0, 0.3 * total, span / 4.0]
    bounds = (
        [0.0, w_lo, span / 40.0, 0.5, 1.5, 0.0, span / 10.0],
        [2.0 * total, w_hi, span / 10.0, 8.0, 25.0, total, span],
    )
    popt, pcov = curve_fit(
        crystalball_plus_exp, centers, counts, p0=p0, sigma=err,
        absolute_sigma=True, bounds=bounds, maxfev=80000,
    )
    perr = np.sqrt(np.diag(pcov))

    residuals = (counts - crystalball_plus_exp(centers, *popt)) / err
    ndf = len(counts) - len(popt)

    return PeakFit(
        name=name,
        window=window,
        mass=float(popt[1]),
        mass_err=float(perr[1]),
        width=float(abs(popt[2])),
        width_err=float(perr[2]),
        amplitude=float(popt[0]),
        chi2_ndf=float(np.sum(residuals**2) / ndf),
        n_events=int(counts.sum()),
        bin_width=float(bin_width),
        beta=float(popt[3]),
        power=float(popt[4]),
        signal_yield=float(popt[0]),
    )
