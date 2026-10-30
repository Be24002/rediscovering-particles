"""Relativistic kinematics of muon pairs.

The invariant mass of two particles is the mass of the system as seen from
its center of mass; for a particle that decays to mu+ mu- it is the mass of
the *parent* particle. That is why plotting m(mu+ mu-) for millions of LHC
collision events makes invisible particles appear as peaks: every Z -> mu+ mu-
decay contributes an entry at 91.19 GeV, whether we saw the Z or not.

    m^2 = (E1 + E2)^2 - |p1 + p2|^2
"""

from __future__ import annotations

import numpy as np

# PDG reference values (2024) -- targets the analysis must reproduce.
MUON_MASS_GEV = 0.105658
JPSI_MASS_GEV = 3.096916
PSI2S_MASS_GEV = 3.686097
UPSILON1S_GEV = 9.4603
Z_MASS_GEV = 91.1876
Z_WIDTH_GEV = 2.4952


def invariant_mass(
    px1, py1, pz1, e1, px2, py2, pz2, e2
):
    """Vectorized invariant mass m(mu1 mu2) from Cartesian 4-momenta."""
    total_e = np.asarray(e1) + np.asarray(e2)
    total_p2 = (
        (np.asarray(px1) + np.asarray(px2)) ** 2
        + (np.asarray(py1) + np.asarray(py2)) ** 2
        + (np.asarray(pz1) + np.asarray(pz2)) ** 2
    )
    m2 = total_e**2 - total_p2
    # Numerical protection: round-off can make m2 slightly negative.
    return np.sqrt(np.clip(m2, 0.0, None))


def spherical_to_cartesian(pt, eta, phi, mass=MUON_MASS_GEV):
    """Convert (pt, eta, phi) to (px, py, pz, E) for a muon.

    px = pt cos(phi),  py = pt sin(phi),  pz = pt sinh(eta),
    E  = sqrt(p^2 + m_mu^2)
    """
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return px, py, pz, e


def invariant_mass_from_kinematics(pt1, eta1, phi1, pt2, eta2, phi2):
    """Invariant mass from (pt, eta, phi) of each muon (NanoAOD-style input)."""
    px1, py1, pz1, e1 = spherical_to_cartesian(pt1, eta1, phi1)
    px2, py2, pz2, e2 = spherical_to_cartesian(pt2, eta2, phi2)
    return invariant_mass(px1, py1, pz1, e1, px2, py2, pz2, e2)
