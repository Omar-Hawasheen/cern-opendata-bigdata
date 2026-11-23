"""Part 2 - A parallel, out-of-core pipeline over 61.5 million CMS collisions.

Input : data/raw/Run2012BC_DoubleMuParked_Muons.root  (CMS Open Data, record 12341,
        2.2 GB, ~61.5M events, NanoAOD-style jagged muon arrays)

Stage 1 - ETL (map):    the event range is split into chunks; a pool of worker
                        processes each read their chunk with uproot, apply the
                        physics selection with vectorised awkward-array ops and
                        write a compressed Parquet part file. No process ever
                        holds the whole dataset in memory.
Stage 2 - Analytics:    Dask reads the Parquet dataset lazily and computes
                        histograms / aggregates in parallel (map -> reduce).

Output: data/processed/dimuons/*.parquet, figures/dimuon_spectrum_2012_61M.png,
        figures/upsilon_region_2012.png, figures/pipeline_cutflow.png,
        results/bigdata_pipeline.json
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
import uproot

from kinematics import RESONANCES, invariant_mass_ptetaphi

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/Run2012BC_DoubleMuParked_Muons.root"
OUT = ROOT / "data/processed/dimuons"
FIG, RES = ROOT / "figures", ROOT / "results"
BRANCHES = ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"]
CHUNK = 2_000_000  # events per task
HIST_BINS = np.logspace(np.log10(0.25), np.log10(300), 1200)


def process_chunk(task):
    """Map step: read one event range, select, compute mass, write Parquet."""
    idx, start, stop = task
    t0 = time.perf_counter()
    with uproot.open(RAW) as f:
        ev = f["Events"].arrays(BRANCHES, entry_start=start, entry_stop=stop)
    n_total = len(ev)

    two = ev[ev.nMuon == 2]                                   # exactly 2 muons
    n_two = len(two)
    os_ = two[two.Muon_charge[:, 0] != two.Muon_charge[:, 1]]  # opposite charge
    n_os = len(os_)

    mu = {k: ak.to_numpy(os_[f"Muon_{k}"]).astype(np.float32)  # (n, 2) regular now
          for k in ("pt", "eta", "phi", "mass")}
    mass = invariant_mass_ptetaphi(
        mu["pt"][:, 0], mu["eta"][:, 0], mu["phi"][:, 0], mu["mass"][:, 0],
        mu["pt"][:, 1], mu["eta"][:, 1], mu["phi"][:, 1], mu["mass"][:, 1],
    ).astype(np.float32)

    table = pa.table({
        "pt1": mu["pt"][:, 0], "eta1": mu["eta"][:, 0], "phi1": mu["phi"][:, 0],
        "pt2": mu["pt"][:, 1], "eta2": mu["eta"][:, 1], "phi2": mu["phi"][:, 1],
        "mass": mass,
    })
    pq.write_table(table, OUT / f"part-{idx:04d}.parquet", compression="zstd")

    return {
        "idx": idx, "n_total": n_total, "n_two_muons": n_two, "n_opposite_charge": n_os,
        "seconds": time.perf_counter() - t0,
        "rss_mb": psutil.Process().memory_info().rss / 2**20,
    }


def run_etl(n_workers):
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.parquet"):
        old.unlink()
    with uproot.open(RAW) as f:
        n_events = f["Events"].num_entries
    tasks = [(i, s, min(s + CHUNK, n_events))
             for i, s in enumerate(range(0, n_events, CHUNK))]
    print(f"{n_events:,} events -> {len(tasks)} chunks on {n_workers} workers")

    t0 = time.perf_counter()
    stats = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(process_chunk, t) for t in tasks]
        for fut in as_completed(futures):
            s = fut.result()
            stats.append(s)
            print(f"  chunk {s['idx']:3d}: {s['n_total']:>9,} ev in {s['seconds']:5.1f}s "
                  f"(worker RSS {s['rss_mb']:.0f} MB)  [{len(stats)}/{len(tasks)}]")
    wall = time.perf_counter() - t0
    return n_events, stats, wall


def run_analytics():
    """Reduce step with Dask: parallel histogramming over the Parquet dataset."""
    import dask
    import dask.dataframe as dd

    t0 = time.perf_counter()
    ddf = dd.read_parquet(str(OUT), columns=["mass", "pt1", "pt2"])

    def part_hist(df):
        h, _ = np.histogram(df["mass"].to_numpy(), bins=HIST_BINS)
        return h

    # one histogram per partition (map), summed (reduce) - never materialises
    # the full column on one machine.
    parts = ddf.to_delayed()
    hists = [dask.delayed(part_hist)(p) for p in parts]
    total = dask.delayed(sum)(hists)
    n_cands = ddf.shape[0]

    windows = {"J/psi": (2.95, 3.25), "Upsilon(1S-3S)": (9.2, 10.6), "Z": (81, 101)}
    counts = {k: ((ddf.mass > lo) & (ddf.mass < hi)).sum() for k, (lo, hi) in windows.items()}
    mean_pt_z = ddf[(ddf.mass > 81) & (ddf.mass < 101)][["pt1", "pt2"]].mean()

    hist, n_cands, counts, mean_pt_z = dask.compute(total, n_cands, counts, mean_pt_z)
    secs = time.perf_counter() - t0
    return hist, int(n_cands), {k: int(v) for k, v in counts.items()}, mean_pt_z, secs


def plot_spectrum(hist, n_events, path):
    fig, ax = plt.subplots(figsize=(11, 6.5))
    centers = np.sqrt(HIST_BINS[1:] * HIST_BINS[:-1])
    widths = np.diff(HIST_BINS)
    ax.step(centers, hist / widths, where="mid", color="#1f4e8c", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 300)
    ax.set_xlabel(r"Dimuon invariant mass $m_{\mu\mu}$ [GeV]")
    ax.set_ylabel(r"Events / GeV")
    ax.set_title(f"CMS 2012 open data: {n_events / 1e6:.1f}M collisions, "
                 "opposite-charge dimuon spectrum")
    ytop = ax.get_ylim()[1]
    for name, m in RESONANCES.items():
        ax.text(m, ytop * 0.35, name, ha="center", fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_upsilon(path):
    """Zoom: the three Upsilon states (bottom-antibottom bound states)."""
    tbl = pq.read_table(OUT, columns=["mass"],
                        filters=[("mass", ">", 8.5), ("mass", "<", 11.5)])
    m = tbl.column("mass").to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(m, bins=150, range=(8.5, 11.5), histtype="stepfilled",
            color="#9ecae1", ec="#1f4e8c")
    for name, mass in {r"$\Upsilon(1S)$": 9.460, r"$\Upsilon(2S)$": 10.023,
                       r"$\Upsilon(3S)$": 10.355}.items():
        ax.axvline(mass, color="#d62728", ls="--", lw=1)
        ax.text(mass, ax.get_ylim()[1] * 0.97, name, ha="center", va="top",
                color="#d62728", bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("Events / 20 MeV")
    ax.set_title(r"Resolving the three $\Upsilon$ states (61.5M-event dataset)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cutflow(cutflow, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels, vals = list(cutflow), list(cutflow.values())
    bars = ax.barh(labels[::-1], vals[::-1], color="#1f4e8c")
    for b, v in zip(bars, vals[::-1]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v / 1e6:.1f}M", va="center")
    ax.set_xlabel("Events")
    ax.set_title("Selection cut-flow")
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    n_workers = max(1, min(8, (os.cpu_count() or 2) - 2))

    n_events, stats, etl_wall = run_etl(n_workers)
    cutflow = {
        "All events": n_events,
        "Exactly 2 muons": sum(s["n_two_muons"] for s in stats),
        "Opposite charge": sum(s["n_opposite_charge"] for s in stats),
    }
    hist, n_cands, window_counts, mean_pt_z, ana_secs = run_analytics()

    plot_spectrum(hist, n_events, FIG / "dimuon_spectrum_2012_61M.png")
    plot_upsilon(FIG / "upsilon_region_2012.png")
    plot_cutflow(cutflow, FIG / "pipeline_cutflow.png")

    raw_gb = RAW.stat().st_size / 1e9
    pq_gb = sum(p.stat().st_size for p in OUT.glob("*.parquet")) / 1e9
    out = {
        "n_events": n_events,
        "workers": n_workers,
        "chunk_size_events": CHUNK,
        "etl_wall_seconds": round(etl_wall, 1),
        "etl_throughput_events_per_sec": round(n_events / etl_wall),
        "max_worker_rss_mb": round(max(s["rss_mb"] for s in stats)),
        "raw_root_gb": round(raw_gb, 2),
        "processed_parquet_gb": round(pq_gb, 3),
        "cutflow": cutflow,
        "dimuon_candidates": n_cands,
        "candidates_in_mass_windows": window_counts,
        "mean_muon_pt_in_Z_window_GeV": {k: round(float(v), 2) for k, v in mean_pt_z.items()},
        "dask_analytics_seconds": round(ana_secs, 2),
    }
    (RES / "bigdata_pipeline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
