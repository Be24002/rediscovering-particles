"""Loading real CMS Open Data dimuon events.

Dataset: CMS Run 2011A "DoubleMu" -- events containing two muons from
pp collisions at sqrt(s) = 7 TeV, released by the CERN Open Data portal
for education. Each row is one muon pair with both muons' 4-momenta,
charges, and the official invariant mass (used here only as a cross-check
of our own calculation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "DoubleMuRun2011A.csv"
DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/cms-opendata-education/"
    "zboson-exercise/master/DoubleMuRun2011A.csv"
)


@dataclass
class DimuonSample:
    """Arrays of per-event quantities for a sample of muon pairs."""

    run: np.ndarray
    event: np.ndarray
    px1: np.ndarray
    py1: np.ndarray
    pz1: np.ndarray
    e1: np.ndarray
    q1: np.ndarray
    px2: np.ndarray
    py2: np.ndarray
    pz2: np.ndarray
    e2: np.ndarray
    q2: np.ndarray
    m_dataset: np.ndarray  # invariant mass published with the dataset

    def __len__(self) -> int:
        return len(self.run)

    @property
    def opposite_charge(self) -> np.ndarray:
        """Mask of mu+ mu- pairs (a resonance decays to opposite charges)."""
        return self.q1 * self.q2 < 0


def download_data(path=DEFAULT_DATA, url=DOWNLOAD_URL) -> Path:
    """Fetch the dataset from the CERN Open Data GitHub mirror (~72 MB)."""
    path = Path(path)
    if path.exists():
        print(f"{path} already exists ({path.stat().st_size / 1e6:.1f} MB)")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def load_dimuon_csv(path=DEFAULT_DATA, max_rows: int | None = None) -> DimuonSample:
    """Parse the CSV into a DimuonSample (vectorized, one pass)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/download_data.py "
            f"(fetches {DOWNLOAD_URL})"
        )
    with path.open() as handle:
        header = handle.readline().strip().split(",")
    # Keep only numeric columns (Type1/Type2 are strings like "G").
    wanted = ["Run", "Event", "E1", "px1", "py1", "pz1", "Q1",
              "E2", "px2", "py2", "pz2", "Q2", "M"]
    usecols = [header.index(name) for name in wanted]
    raw = np.loadtxt(
        path, delimiter=",", skiprows=1, unpack=True,
        usecols=usecols, max_rows=max_rows,
    )
    cols = dict(zip(wanted, raw))
    return DimuonSample(
        run=cols["Run"].astype(np.int64),
        event=cols["Event"].astype(np.int64),
        px1=cols["px1"], py1=cols["py1"], pz1=cols["pz1"],
        e1=cols["E1"], q1=cols["Q1"],
        px2=cols["px2"], py2=cols["py2"], pz2=cols["pz2"],
        e2=cols["E2"], q2=cols["Q2"],
        m_dataset=cols["M"],
    )
