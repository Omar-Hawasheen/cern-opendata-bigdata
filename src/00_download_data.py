"""Download the CERN Open Data files used by this project into data/raw/.

    python src/00_download_data.py           # small files only (~80 MB)
    python src/00_download_data.py --all     # + the 2.2 GB CMS 2012 ROOT file
"""
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
BASE = "https://opendata.cern.ch/eos/opendata"
FILES = {
    # CMS 2011 DoubleMu dimuon events - record 545
    "Dimuon_DoubleMu.csv": f"{BASE}/cms/Run2011A/DoubleMu/CSV/12Oct2013-v1/Dimuon_DoubleMu.csv",
    # ATLAS Higgs ML challenge 2014 - record 328
    "atlas-higgs-challenge-2014-v2.csv.gz":
        f"{BASE}/atlas/higgs-challenge-2014/atlas-higgs-challenge-2014-v2.csv.gz",
}
BIG = {
    # CMS 2012 DoubleMuParked, NanoAOD reduced on muons - record 12341
    "Run2012BC_DoubleMuParked_Muons.root":
        f"{BASE}/cms/derived-data/AOD2NanoAODOutreachTool/Run2012BC_DoubleMuParked_Muons.root",
}


def download(name, url):
    dest = RAW / name
    if dest.exists():
        print(f"skip {name} (exists)")
        return
    print(f"downloading {name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB", end="")
    print()
    tmp.rename(dest)


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    todo = {**FILES, **(BIG if "--all" in sys.argv else {})}
    for n, u in todo.items():
        download(n, u)
