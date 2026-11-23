"""Part 2b - The same physics analysis in Apache Spark SQL.

Input : data/processed/dimuons/*.parquet (written by 02_bigdata_pipeline.py from
        the 61.5M-event CMS ROOT file). Spark cannot read ROOT natively, so
        ROOT -> Parquet is the bridge into the standard big data ecosystem.

Everything below is plain SQL, so the same queries run unchanged on a
multi-node Spark cluster, Databricks, or CERN's own Spark service (SWAN / NXCALS).

Output: figures/spark_*.png, results/spark_sql.json
"""
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession

from kinematics import RESONANCES

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/dimuons"
FIG = Path(os.environ.get("PIPELINE_FIG", ROOT / "figures"))
RES = Path(os.environ.get("PIPELINE_RES", ROOT / "results"))

LO, HI, NBINS = np.log10(0.25), np.log10(300), 600
W = (HI - LO) / NBINS

QUERIES = {
    "overview": """
        SELECT COUNT(*)              AS candidates,
               ROUND(MIN(mass), 3)   AS min_mass,
               ROUND(MAX(mass), 1)   AS max_mass,
               ROUND(AVG(pt1 + pt2), 2) AS avg_scalar_pt_sum
        FROM dimuons""",

    # log-spaced histogram computed entirely inside Spark (GROUP BY bin index)
    "histogram": f"""
        SELECT CAST(FLOOR((LOG10(mass) - {LO}) / {W}) AS INT) AS bin,
               COUNT(*) AS n
        FROM dimuons
        WHERE mass > 0.25 AND mass < 300
        GROUP BY bin
        ORDER BY bin""",

    "resonance_windows": """
        SELECT CASE
                 WHEN mass BETWEEN 0.75  AND 0.82  THEN '1 rho/omega'
                 WHEN mass BETWEEN 0.99  AND 1.05  THEN '2 phi'
                 WHEN mass BETWEEN 2.95  AND 3.25  THEN '3 J/psi'
                 WHEN mass BETWEEN 3.60  AND 3.78  THEN '4 psi(2S)'
                 WHEN mass BETWEEN 9.20  AND 10.60 THEN '5 Upsilon(1S,2S,3S)'
                 WHEN mass BETWEEN 81    AND 101   THEN '6 Z boson'
               END AS resonance,
               COUNT(*) AS candidates
        FROM dimuons
        GROUP BY resonance
        HAVING resonance IS NOT NULL
        ORDER BY resonance""",

    # detector physics: muons in the central barrel are measured better than in
    # the forward endcaps -> the Z peak should be narrower for barrel-barrel pairs
    "z_resolution_by_detector_region": """
        WITH z AS (
          SELECT mass,
                 CASE
                   WHEN ABS(eta1) < 1.2 AND ABS(eta2) < 1.2 THEN 'barrel-barrel'
                   WHEN ABS(eta1) >= 1.2 AND ABS(eta2) >= 1.2 THEN 'endcap-endcap'
                   ELSE 'barrel-endcap'
                 END AS region
          FROM dimuons
          WHERE mass BETWEEN 86 AND 96
        )
        SELECT region,
               COUNT(*)                         AS candidates,
               ROUND(AVG(mass), 3)              AS mean_mass,
               ROUND(STDDEV(mass), 3)           AS stddev_mass,
               ROUND(PERCENTILE_APPROX(mass, 0.5), 3) AS median_mass
        FROM z
        GROUP BY region
        ORDER BY stddev_mass""",

    "highest_mass_events": """
        SELECT ROUND(mass, 1) AS mass, ROUND(pt1, 1) AS pt1, ROUND(pt2, 1) AS pt2,
               ROUND(eta1, 2) AS eta1, ROUND(eta2, 2) AS eta2
        FROM dimuons
        ORDER BY mass DESC
        LIMIT 5""",
}


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = (SparkSession.builder
             .appName("cms-dimuon-sparksql")
             .master("local[*]")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "32")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    t_all = time.perf_counter()
    # Pass part files explicitly: on Windows without Hadoop native libraries
    # (winutils), Spark fails when listing a local directory. On Linux or a
    # cluster you'd simply use spark.read.parquet(str(DATA)) or an s3:// path.
    files = [p.as_posix() for p in sorted(DATA.glob("*.parquet"))]
    spark.read.parquet(*files).createOrReplaceTempView("dimuons")

    results, timings = {}, {}
    for name, sql in QUERIES.items():
        t0 = time.perf_counter()
        rows = [r.asDict() for r in spark.sql(sql).collect()]
        timings[name] = round(time.perf_counter() - t0, 2)
        results[name] = rows
        if name != "histogram":
            print(f"\n-- {name} ({timings[name]} s)")
            for r in rows:
                print("  ", r)

    # --- figures -------------------------------------------------------------
    edges = 10 ** (LO + W * np.arange(NBINS + 1))
    counts = np.zeros(NBINS)
    for r in results["histogram"]:
        if 0 <= r["bin"] < NBINS:
            counts[r["bin"]] = r["n"]
    centers = np.sqrt(edges[1:] * edges[:-1])

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.step(centers, counts / np.diff(edges), where="mid", color="#e25a1c", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.25, 300)
    ax.set_xlabel(r"Dimuon invariant mass $m_{\mu\mu}$ [GeV]")
    ax.set_ylabel("Events / GeV")
    n_cand = results["overview"][0]["candidates"]
    ax.set_title(f"Histogram computed by Spark SQL ({n_cand / 1e6:.1f}M dimuon candidates)")
    ytop = ax.get_ylim()[1]
    for name, m in RESONANCES.items():
        ax.text(m, ytop * 0.35, name, ha="center", fontsize=12,
                bbox=dict(fc="white", ec="none", alpha=0.8))
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "spark_dimuon_spectrum.png", dpi=150)
    plt.close(fig)

    reg = results["z_resolution_by_detector_region"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["region"] for r in reg], [r["stddev_mass"] for r in reg],
           color=["#2ca02c", "#ff7f0e", "#d62728"][:len(reg)])
    for i, r in enumerate(reg):
        ax.text(i, r["stddev_mass"], f"{r['stddev_mass']:.2f} GeV\n({r['candidates']:,})",
                ha="center", va="bottom")
    ax.set_ylabel("Spread of Z peak, 86-96 GeV [GeV]")
    ax.set_title("Detector resolution by region (Spark SQL)")
    ax.set_ylim(0, max(r["stddev_mass"] for r in reg) * 1.3)
    fig.tight_layout()
    fig.savefig(FIG / "spark_z_resolution_by_region.png", dpi=150)
    plt.close(fig)

    out = {
        "spark_version": spark.version,
        "master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "query_seconds": timings,
        "total_seconds": round(time.perf_counter() - t_all, 2),
        **{k: v for k, v in results.items() if k != "histogram"},
    }
    (RES / "spark_sql.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("spark_version", "query_seconds", "total_seconds")},
                     indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
