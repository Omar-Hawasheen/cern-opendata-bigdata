"""Relativistic kinematics helpers (natural units, GeV, c = 1)."""
import numpy as np

MUON_MASS = 0.1056583755  # GeV


def invariant_mass(E1, px1, py1, pz1, E2, px2, py2, pz2):
    """Invariant mass of a two-particle system from four-momenta.

    M^2 = (E1 + E2)^2 - |p1 + p2|^2
    """
    E = E1 + E2
    px, py, pz = px1 + px2, py1 + py2, pz1 + pz2
    m2 = E**2 - (px**2 + py**2 + pz**2)
    return np.sqrt(np.clip(m2, 0, None))


def invariant_mass_ptetaphi(pt1, eta1, phi1, m1, pt2, eta2, phi2, m2):
    """Invariant mass from detector coordinates (pt, eta, phi, m).

    Uses the collider form:
    M^2 = m1^2 + m2^2 + 2 (E1 E2 - p1 . p2), with
    E = sqrt(m^2 + (pt cosh eta)^2),  p . p' = pt pt' (cosh(eta - eta') ... )
    expressed here via explicit Cartesian components for clarity.
    """
    px1, py1, pz1 = pt1 * np.cos(phi1), pt1 * np.sin(phi1), pt1 * np.sinh(eta1)
    px2, py2, pz2 = pt2 * np.cos(phi2), pt2 * np.sin(phi2), pt2 * np.sinh(eta2)
    E1 = np.sqrt(m1**2 + px1**2 + py1**2 + pz1**2)
    E2 = np.sqrt(m2**2 + px2**2 + py2**2 + pz2**2)
    return invariant_mass(E1, px1, py1, pz1, E2, px2, py2, pz2)


# Known resonances decaying to mu+ mu- (PDG masses, GeV), used to label plots.
RESONANCES = {
    r"$\eta$": 0.548,
    r"$\rho,\omega$": 0.78,
    r"$\phi$": 1.019,
    r"$J/\psi$": 3.097,
    r"$\psi(2S)$": 3.686,
    r"$\Upsilon$": 9.46,
    r"$Z$": 91.19,
}
