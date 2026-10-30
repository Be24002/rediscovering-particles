"""Fit the J/psi and Z peaks; compare measured masses with PDG values.

The Z is fitted with two models to show how the verdict depends on physics
modeling, not just statistics:

1. Gaussian + linear background (naive: symmetric resolution, flat background)
2. Crystal Ball + exponential background (physical: low-mass bremsstrahlung
   tail, smoothly falling Drell-Yan background)

Produces figures/Jpsi_peak_fit.png, figures/Z_peak_fit.png and
results/peak_fits.csv.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import crystalball

from dimuon.data import DEFAULT_DATA, load_dimuon_csv
from dimuon.fitting import fit_peak, fit_peak_asymmetric
from dimuon.physics import JPSI_MASS_GEV, Z_MASS_GEV, invariant_mass

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("Loading CMS Run 2011A DoubleMu events ...")
sample = load_dimuon_csv(DEFAULT_DATA)
masses = invariant_mass(
    sample.px1, sample.py1, sample.pz1, sample.e1,
    sample.px2, sample.py2, sample.pz2, sample.e2,
)
m = masses[sample.opposite_charge]

jpsi = fit_peak(m, (2.85, 3.40), name="J/psi (Gaussian)", bins=70)
z_gauss = fit_peak(m, (70.0, 110.0), name="Z (Gaussian)", bins=80)
z_cb = fit_peak_asymmetric(m, (70.0, 110.0), name="Z (CrystalBall)", bins=80)

fits = [(jpsi, JPSI_MASS_GEV), (z_gauss, Z_MASS_GEV), (z_cb, Z_MASS_GEV)]
for fit, pdg_mass in fits:
    delta = fit.mass - pdg_mass
    tail = (
        f" | tail: beta = {fit.beta:.2f}, power = {fit.power:.2f}"
        if fit.beta
        else ""
    )
    print(
        f"{fit.name:>18}: m = {fit.mass:.3f} +- {fit.mass_err:.3f} GeV "
        f"(PDG {pdg_mass:.4f}, diff {delta:+.3f} GeV, "
        f"{abs(delta) / fit.mass_err:.1f} sigma_stat) | "
        f"sigma = {fit.width:.3f} GeV | chi2/ndf = {fit.chi2_ndf:.2f} | "
        f"~{fit.n_signal:,.0f} signal events" + tail
    )

# ---- Figure 1: J/psi ------------------------------------------------------
nbins, window = 70, (2.85, 3.40)
counts, edges = np.histogram(m, bins=nbins, range=window)
centers = 0.5 * (edges[1:] + edges[:-1])
signal = jpsi.amplitude * np.exp(-0.5 * ((centers - jpsi.mass) / jpsi.width) ** 2)

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.errorbar(centers, counts, yerr=np.sqrt(counts + 1), fmt=".", color="k",
            label="CMS Open Data (Run 2011A)")
ax.plot(centers, signal, "-", color="C3",
        label=f"Gaussian: $m = {jpsi.mass:.3f} \\pm {jpsi.mass_err:.3f}$ GeV")
ax.set_xlabel("invariant mass  $m(\\mu^+\\mu^-)$  [GeV]")
ax.set_ylabel(f"events / {edges[1] - edges[0]:.4f} GeV")
ax.set_title(f"J/$\\psi$ peak: measured {jpsi.mass:.3f} GeV vs "
             f"PDG {JPSI_MASS_GEV:.4f} GeV ($\\chi^2/ndf = {jpsi.chi2_ndf:.2f}$)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "Jpsi_peak_fit.png", dpi=150)
plt.close(fig)

# ---- Figure 2: Z with both models ----------------------------------------
nbins, window = 80, (70.0, 110.0)
counts, edges = np.histogram(m, bins=nbins, range=window)
centers = 0.5 * (edges[1:] + edges[:-1])
bin_width = edges[1] - edges[0]
gauss_curve = z_gauss.amplitude * np.exp(
    -0.5 * ((centers - z_gauss.mass) / z_gauss.width) ** 2
)
cb_curve = (
    z_cb.n_signal
    * bin_width
    * crystalball.pdf(centers, z_cb.beta, z_cb.power, loc=z_cb.mass, scale=z_cb.width)
)

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.errorbar(centers, counts, yerr=np.sqrt(counts + 1), fmt=".", color="k",
            ms=3, label="CMS Open Data (Run 2011A)")
ax.plot(centers, gauss_curve, "-", color="C0",
        label=f"Gaussian: $m = {z_gauss.mass:.2f} \\pm {z_gauss.mass_err:.2f}$ GeV")
ax.plot(centers, cb_curve, "-", color="C3",
        label=(f"Crystal Ball: $m = {z_cb.mass:.2f} \\pm {z_cb.mass_err:.2f}$ GeV "
               f"(tail $\\beta = {z_cb.beta:.1f}\\sigma$)"))
ax.axvline(Z_MASS_GEV, color="k", ls=":", lw=1,
           label=f"PDG $m_Z = {Z_MASS_GEV:.2f}$ GeV")
ax.set_xlabel("invariant mass  $m(\\mu^+\\mu^-)$  [GeV]")
ax.set_ylabel(f"events / {bin_width:.2f} GeV")
ax.set_title("Z peak: the bremsstrahlung tail pulls the naive Gaussian low")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGURES / "Z_peak_fit.png", dpi=150)
plt.close(fig)

# ---- CSV -------------------------------------------------------------------
with (RESULTS / "peak_fits.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["model", "mass_gev", "mass_err_gev", "width_gev",
                     "width_err_gev", "chi2_ndf", "n_events_window",
                     "n_signal", "pdg_mass_gev", "diff_gev"])
    for fit, pdg_mass in fits:
        writer.writerow([fit.name, f"{fit.mass:.4f}", f"{fit.mass_err:.4f}",
                         f"{fit.width:.4f}", f"{fit.width_err:.4f}",
                         f"{fit.chi2_ndf:.2f}", fit.n_events,
                         f"{fit.n_signal:.0f}", f"{pdg_mass:.4f}",
                         f"{fit.mass - pdg_mass:+.4f}"])
print("Wrote figures/Jpsi_peak_fit.png, figures/Z_peak_fit.png, results/peak_fits.csv")
