"""The money plot: the dimuon invariant-mass spectrum of real LHC data.

Every peak is a particle rediscovered: J/psi, psi(2S), the Upsilon family,
and the Z boson. Produces figures/dimuon_spectrum.png and
results/dimuon_spectrum.csv.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dimuon.data import DEFAULT_DATA, load_dimuon_csv
from dimuon.physics import (
    JPSI_MASS_GEV,
    UPSILON1S_GEV,
    Z_MASS_GEV,
    invariant_mass,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

print("Loading CMS Run 2011A DoubleMu events ...")
sample = load_dimuon_csv(DEFAULT_DATA)
n_all = len(sample)

masses = invariant_mass(
    sample.px1, sample.py1, sample.pz1, sample.e1,
    sample.px2, sample.py2, sample.pz2, sample.e2,
)

# Cross-check: our kinematics vs the mass published with the dataset.
delta = np.abs(masses - sample.m_dataset)
print(f"Events loaded                      : {n_all}")
print(f"Oppposite-charge (mu+ mu-) pairs   : {int(sample.opposite_charge.sum())}")
print(f"median |m_ours - m_dataset| (GeV)  : {np.median(delta):.2e}  (rounding level)")
print(f"99.9% quantile of |dm| (GeV)       : {np.percentile(delta, 99.9):.4f}")
print(f"outlier rows (>50 MeV)             : {int((delta > 0.05).sum())} of {len(delta)}")

m = masses[sample.opposite_charge]

# ---- Spectrum ------------------------------------------------------------
bins = np.logspace(np.log10(0.5), np.log10(130.0), 250)
counts, edges = np.histogram(m, bins=bins)
centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean for log bins

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.fill_between(centers, counts, step="mid", color="C0", alpha=0.6)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.5, 130)
ax.set_ylim(bottom=0.7)

peaks = [
    (JPSI_MASS_GEV, "J/$\\psi$"),
    (3.686, "$\\psi$(2S)"),
    (UPSILON1S_GEV, "$\\Upsilon$(1S,2S,3S)"),
    (Z_MASS_GEV, "Z boson"),
]
for mass_gev, label in peaks:
    ax.axvline(mass_gev, color="k", ls=":", lw=0.8)
    ax.annotate(
        label, xy=(mass_gev, ax.get_ylim()[1]),
        xytext=(mass_gev * 0.92, np.sqrt(ax.get_ylim()[0] * ax.get_ylim()[1]) * 30),
        fontsize=11, rotation=90, va="top",
    )

ax.set_xlabel("invariant mass  $m(\\mu^+\\mu^-)$  [GeV]")
ax.set_ylabel("events / bin")
ax.set_title(
    f"CMS Open Data, $\\sqrt{{s}} = 7$ TeV (Run 2011A): "
    f"{n_all:,} dimuon events, {len(m):,} opposite-charge"
)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(FIGURES / "dimuon_spectrum.png", dpi=150)
plt.close(fig)

np.savetxt(
    RESULTS / "dimuon_spectrum.csv",
    np.column_stack([centers, counts]),
    delimiter=",",
    header="mass_gev_geometric_center,events_per_bin",
    comments="",
)
print("Wrote figures/dimuon_spectrum.png, results/dimuon_spectrum.csv")
