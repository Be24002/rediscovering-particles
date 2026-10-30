"""Physics regression tests: kinematics by hand, dataset cross-check,
fit recovery of an injected peak, and the NanoAOD loader round-trip."""

from pathlib import Path

import numpy as np
import pytest

from dimuon.data import DEFAULT_DATA, DimuonSample, load_dimuon_csv
from dimuon.fitting import fit_peak
from dimuon.physics import (
    JPSI_MASS_GEV,
    MUON_MASS_GEV,
    Z_MASS_GEV,
    invariant_mass,
    invariant_mass_from_kinematics,
    spherical_to_cartesian,
)

# ------------------------------------------------------------- kinematics


def test_invariant_mass_of_two_muons_at_rest():
    # Two muons at rest: m = 2 * m_mu, exactly.
    m = invariant_mass(0, 0, 0, MUON_MASS_GEV, 0, 0, 0, MUON_MASS_GEV)
    assert m == pytest.approx(2 * MUON_MASS_GEV, abs=1e-12)


def test_back_to_back_pair_mass_is_total_energy():
    # p1 = (E, 0, 0, p), p2 = (E, 0, 0, -p) -> total momentum zero, m = 2E.
    e, p = 100.0, 99.9999
    m = invariant_mass(0, 0, p, e, 0, 0, -p, e)
    assert m == pytest.approx(200.0, abs=1e-9)


def test_pt_eta_phi_route_agrees_with_cartesian():
    # The same muon pair expressed two ways must give the same mass.
    # Pair: muon A = (pt, eta, phi), muon B = (pt, eta, phi + pi) -- the two
    # muons are back-to-back in azimuth and parallel along z, so pz does not
    # cancel and m = 2*sqrt(pt^2 + m_mu^2) analytically.
    pt, eta, phi = 45.0, 1.2, 0.7
    px, py, pz, energy = spherical_to_cartesian(pt, eta, phi)
    m_a = invariant_mass_from_kinematics(pt, eta, phi, pt, eta, phi + np.pi)
    m_b = invariant_mass(px, py, pz, energy, -px, -py, pz, energy)
    assert m_a == pytest.approx(m_b, rel=1e-9)
    assert m_a == pytest.approx(2 * np.sqrt(pt**2 + MUON_MASS_GEV**2), rel=1e-9)


# ------------------------------------------------------------------- data


def test_opposite_charge_selection():
    sample = DimuonSample(
        run=np.array([1, 2, 3]), event=np.array([1, 2, 3]),
        px1=np.zeros(3), py1=np.zeros(3), pz1=np.zeros(3),
        e1=np.ones(3), q1=np.array([1, 1, -1]),
        px2=np.zeros(3), py2=np.zeros(3), pz2=np.zeros(3),
        e2=np.ones(3), q2=np.array([-1, 1, -1]),
        m_dataset=np.zeros(3),
    )
    assert list(sample.opposite_charge) == [True, False, False]


@pytest.mark.skipif(not DEFAULT_DATA.exists(), reason="dataset not downloaded")
def test_our_mass_matches_the_published_column():
    # Our kinematics vs the mass published with the CMS dataset: typical
    # events agree at the CSV rounding level (~1e-4 GeV). A handful of rows
    # (~0.05%) differ more -- catastrophic cancellation amplifies the
    # 6-significant-digit rounding for near-identical 4-vectors, and the file
    # contains a few badly reconstructed events (e.g. an 8.7 TeV muon at a
    # 7 TeV collider). Real analyses apply quality cuts for exactly this
    # reason, so we assert robust statistics rather than the max.
    sample = load_dimuon_csv(DEFAULT_DATA, max_rows=20000)
    m = invariant_mass(
        sample.px1, sample.py1, sample.pz1, sample.e1,
        sample.px2, sample.py2, sample.pz2, sample.e2,
    )
    delta = np.abs(m - sample.m_dataset)
    assert np.median(delta) < 1e-3            # typical event: rounding-level match
    assert np.percentile(delta, 99.9) < 0.02  # all but 0.1% agree within 20 MeV
    assert (delta > 0.05).mean() < 0.005      # outliers are rare (<0.5%)


# --------------------------------------------------------------- fitting


def test_fit_recovers_injected_z_peak():
    # Inject a known Gaussian peak on a linear background; the fitter must
    # return the injected mass and width.
    rng = np.random.default_rng(7)
    signal = rng.normal(91.1876, 2.0, size=30_000)
    background = rng.uniform(70.0, 110.0, size=20_000)
    fit = fit_peak(
        np.concatenate([signal, background]), (70.0, 110.0), name="injected"
    )
    assert fit.mass == pytest.approx(91.1876, abs=0.05)
    assert fit.width == pytest.approx(2.0, rel=0.1)


def test_fit_recovers_injected_jpsi_peak():
    rng = np.random.default_rng(11)
    signal = rng.normal(JPSI_MASS_GEV, 0.03, size=40_000)
    background = rng.uniform(2.85, 3.40, size=30_000)
    fit = fit_peak(
        np.concatenate([signal, background]), (2.85, 3.40), name="injected"
    )
    assert fit.mass == pytest.approx(JPSI_MASS_GEV, abs=0.01)


def test_asymmetric_fit_recovers_injected_z_with_tail():
    # Inject a Crystal Ball (Gaussian core + low-mass tail, like real Z
    # events after bremsstrahlung) and check the asymmetric fitter gets
    # both the mass and the core width back.
    from scipy.stats import crystalball

    from dimuon.fitting import fit_peak_asymmetric

    rng = np.random.default_rng(21)
    signal = crystalball.rvs(
        beta=1.2, m=3.0, loc=91.19, scale=2.0, size=40_000, random_state=rng
    )
    background = rng.uniform(70.0, 110.0, size=25_000)
    fit = fit_peak_asymmetric(
        np.concatenate([signal, background]), (70.0, 110.0), name="injected"
    )
    assert fit.mass == pytest.approx(91.19, abs=0.06)
    assert fit.width == pytest.approx(2.0, rel=0.15)


def test_pdg_constants_are_sane():
    # Guards against typos in the reference values.
    assert Z_MASS_GEV == pytest.approx(91.1876, abs=1e-4)
    assert JPSI_MASS_GEV == pytest.approx(3.0969, abs=1e-3)


# ---------------------------------------------------------------- NanoAOD


def test_nanoaod_loader_roundtrip(tmp_path: Path):
    uproot = pytest.importorskip("uproot")
    ak = pytest.importorskip("awkward")
    from dimuon.nanoaod import load_dimuon_from_root

    # Write a tiny NanoAOD-shaped file: 3 events, middle one has 2 muons.
    path = tmp_path / "nano.root"
    with uproot.recreate(path) as f:
        f["Events"] = {
            "Muon_pt": ak.Array([[25.0, 30.0], [40.0], [10.0, 20.0]]),
            "Muon_eta": ak.Array([[0.5, -0.5], [1.0], [0.0, 0.0]]),
            "Muon_phi": ak.Array([[0.0, np.pi], [0.0], [0.3, 0.3 + np.pi]]),
            "Muon_charge": ak.Array([[-1, 1], [1], [1, -1]]),
        }

    events = load_dimuon_from_root(path)
    assert len(events["m"]) == 2  # only exactly-two-muon events survive

    expected = invariant_mass_from_kinematics(
        25.0, 0.5, 0.0, 30.0, -0.5, np.pi
    )
    assert events["m"][0] == pytest.approx(expected, rel=1e-9)
    assert events["q1"][0] * events["q2"][0] < 0
