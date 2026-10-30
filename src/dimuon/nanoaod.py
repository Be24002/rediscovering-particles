"""Read dimuon events directly from CMS NanoAOD ROOT files.

NanoAOD is the compact analysis format of CMS: one ROOT TTree called
``Events`` with per-event jagged arrays of physics objects. Reading it with
uproot + awkward is how a modern Python-based CMS analysis begins -- no ROOT
installation required.

Here we select events with exactly two muons and pair them, computing the
same invariant-mass observable as in the CSV path. A unit test round-trips a
tiny NanoAOD-shaped file written with uproot itself, so the machinery is
verified without depending on the network.
"""

from __future__ import annotations

import numpy as np

try:
    import awkward as ak
    import uproot

    HAVE_UPROOT = True
except ImportError:  # pragma: no cover - extra not installed
    HAVE_UPROOT = False

from .physics import invariant_mass_from_kinematics


def load_dimuon_from_root(source) -> dict:
    """Load exactly-two-muon events from a NanoAOD file or URL.

    Returns dict with pt1/eta1/phi1/q1, pt2/eta2/phi2/q2, and the pair
    invariant mass ``m`` (GeV).
    """
    if not HAVE_UPROOT:
        raise ImportError(
            "uproot/awkward not installed. Run: pip install 'dimuon[nanoaod]'"
        )
    tree = uproot.open(source)["Events"]
    arrays = tree.arrays(
        ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge"], library="ak"
    )
    two = arrays[ak.num(arrays["Muon_pt"]) == 2]
    return {
        "pt1": np.asarray(two["Muon_pt"][:, 0]),
        "eta1": np.asarray(two["Muon_eta"][:, 0]),
        "phi1": np.asarray(two["Muon_phi"][:, 0]),
        "q1": np.asarray(two["Muon_charge"][:, 0], dtype=float),
        "pt2": np.asarray(two["Muon_pt"][:, 1]),
        "eta2": np.asarray(two["Muon_eta"][:, 1]),
        "phi2": np.asarray(two["Muon_phi"][:, 1]),
        "q2": np.asarray(two["Muon_charge"][:, 1], dtype=float),
        "m": invariant_mass_from_kinematics(
            np.asarray(two["Muon_pt"][:, 0]), np.asarray(two["Muon_eta"][:, 0]),
            np.asarray(two["Muon_phi"][:, 0]),
            np.asarray(two["Muon_pt"][:, 1]), np.asarray(two["Muon_eta"][:, 1]),
            np.asarray(two["Muon_phi"][:, 1]),
        ),
    }
