"""dimuon: rediscover particles in real CMS Open Data.

Modules
-------
physics
    Relativistic two-body kinematics (invariant mass) and PDG reference values.
data
    Loader for the CMS Run 2011A DoubleMu dimuon dataset (CSV) and selections.
fitting
    Gaussian + linear background peak fits with proper statistical errors.
nanoaod
    Same analysis objects read straight from CMS NanoAOD ROOT files
    (uproot + awkward), the format used by real CMS analyses.
"""

__version__ = "0.1.0"
