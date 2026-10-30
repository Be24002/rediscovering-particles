"""Fetch the CMS Open Data dimuon dataset (~72 MB) if not present."""

from pathlib import Path

from dimuon.data import download_data

if __name__ == "__main__":
    path = download_data()
    print(f"Ready: {path}")
