"""Part 1 - Rediscovering particles in CMS 2011 dimuon data.

Input : data/raw/Dimuon_DoubleMu.csv  (CMS Open Data, record 545, ~100k events)
Output: figures/dimuon_spectrum_2011.png, figures/z_peak_fit.png,
        figures/jpsi_peak_fit.png, results/dimuon_2011.json
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

from kinematics import RESONANCES, invariant_mass

ROOT = Path(__file__).resolve().parent.parent
RAW, FIG, RES = ROOT / "data/raw", ROOT / "figures", ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)


def load():
    df = pd.read_csv(RAW / "Dimuon_DoubleMu.csv")
    # Recompute the invariant mass ourselves from the four-momenta and check
    # it against the value CMS stored in the file.
    df["M_ours"] = invariant_mass(df.E1, df.px1, df.py1, df.pz1,
                                  df.E2, df.px2, df.py2, df.pz2)
    return df


def plot_spectrum(m, title, path, label_min=0.0):
    bins = np.logspace(np.log10(max(m.min(), 0.25)), np.log10(m.max()), 400)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(m, bins=bins, histtype="step", color="#1f4e8c", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimuon invariant mass $m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_title(title)
    ymax = ax.get_ylim()[1]
    shown = [(n, m) for n, m in RESONANCES.items()
             if bins[0] < m < bins[-1] and m >= label_min]
    for i, (name, mass) in enumerate(shown):
        ax.axvline(mass, color="grey", ls=":", lw=0.8)
        ax.text(mass, ymax * (0.5 if i % 2 == 0 else 0.2), name, ha="center", fontsize=11,
                    bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --- peak models -------------------------------------------------------------
def voigt_plus_bkg(x, n, mu, sigma, gamma, a, b):
    """Z lineshape: Breit-Wigner (natural width) convolved with Gaussian
    (detector resolution) = Voigt profile, on a linear background."""
    return n * voigt_profile(x - mu, sigma, gamma / 2) + a + b * (x - mu)


def gauss_plus_bkg(x, n, mu, sigma, a, b):
    return n * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + a + b * (x - mu)


def fit_peak(m, lo, hi, nbins, model, p0, path, title, xlabel_note=""):
    counts, edges = np.histogram(m[(m > lo) & (m < hi)], bins=nbins, range=(lo, hi))
    x = 0.5 * (edges[1:] + edges[:-1])
    err = np.sqrt(np.maximum(counts, 1))
    popt, pcov = curve_fit(model, x, counts, p0=p0, sigma=err, absolute_sigma=True,
                           maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    chi2 = float(np.sum(((counts - model(x, *popt)) / err) ** 2))
    ndf = len(x) - len(popt)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, counts, yerr=err, fmt="o", ms=3, color="k", label="CMS Open Data")
    xx = np.linspace(lo, hi, 600)
    ax.plot(xx, model(xx, *popt), color="#d62728", lw=2, label="Fit")
    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]" + xlabel_note)
    ax.set_ylabel(f"Events / {(hi - lo) / nbins:.3g} GeV")
    ax.set_title(title)
    ax.text(0.03, 0.95,
            f"$m$ = {popt[1]:.3f} $\\pm$ {perr[1]:.3f} GeV\n"
            f"$\\sigma$ = {abs(popt[2]):.3f} GeV\n"
            f"$\\chi^2$/ndf = {chi2:.1f}/{ndf}",
            transform=ax.transAxes, va="top",
            bbox=dict(fc="white", ec="grey", alpha=0.9))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return popt, perr, chi2, ndf


def main():
    df = load()
    diff = (df.M - df.M_ours).abs()
    print(f"events: {len(df):,}")
    print(f"max |M_cms - M_ours| = {diff.max():.4f} GeV (median {diff.median():.2e})")

    plot_spectrum(df.M_ours.values,
                  "CMS 2011 open data (DoubleMu, ~100k events): dimuon mass spectrum",
                  FIG / "dimuon_spectrum_2011.png", label_min=0.7)

    # Z boson: PDG mass 91.1876 GeV, width 2.4952 GeV
    pz, ez, chi2z, ndfz = fit_peak(
        df.M_ours.values, 70, 110, 80, voigt_plus_bkg,
        p0=[len(df) * 0.1, 91, 2, 2.5, 10, 0], path=FIG / "z_peak_fit.png",
        title="Z boson peak - Voigt fit (Breit-Wigner $\\otimes$ Gaussian)")
    # J/psi: PDG 3.0969 GeV
    pj, ej, chi2j, ndfj = fit_peak(
        df.M_ours.values, 2.8, 3.4, 60, gauss_plus_bkg,
        p0=[500, 3.1, 0.03, 50, 0], path=FIG / "jpsi_peak_fit.png",
        title=r"J/$\psi$ meson peak - Gaussian fit")

    out = {
        "n_events": int(len(df)),
        "mass_recompute_max_abs_diff_GeV": float(diff.max()),
        "Z": {"mass": pz[1], "mass_err": ez[1], "gamma_fit": abs(pz[3]),
              "resolution_sigma": abs(pz[2]), "chi2": chi2z, "ndf": ndfz,
              "pdg_mass": 91.1876, "pdg_width": 2.4952},
        "Jpsi": {"mass": pj[1], "mass_err": ej[1], "resolution_sigma": abs(pj[2]),
                 "chi2": chi2j, "ndf": ndfj, "pdg_mass": 3.0969},
    }
    (RES / "dimuon_2011.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
