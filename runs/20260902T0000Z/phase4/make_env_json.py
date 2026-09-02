#!/usr/bin/env python3
"""Write the environment and Phase-2 facts of this campaign as fresh values
keyed by canonical quantity keys, for build_fresh_values.py --env-json.

Reads live version strings from the venv and nvidia-smi, the cgroup memory
limit, the BigQuery count CSV and the release facts recorded in the run
directory, and writes env.json (qkey -> {value, unit, artifact, command,
utc, note}).

Usage:
    make_env_json.py --run-dir DIR --out env.json
"""

import argparse
import csv
import json
import platform
import subprocess
import time

import cupy
import numpy
import polars


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec = lambda v, u, art, cmd, note="": {"value": v, "unit": u, "artifact": art, "command": cmd, "utc": now, "note": note}  # noqa: E731
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    names = sorted({l.split(",")[0].strip() for l in smi})
    vram = round(max(int(l.split(",")[1].strip().split()[0]) for l in smi) / 1024)
    driver = smi[0].split(",")[2].strip()
    nvcc = subprocess.run(["bash", "-c", "nvcc --version | grep -oE 'release [0-9.]+' | head -1"], capture_output=True, text=True).stdout.strip()
    os_name = subprocess.run(["bash", "-c", ". /etc/os-release && echo $PRETTY_NAME"], capture_output=True, text=True).stdout.strip()
    cg = int(open("/sys/fs/cgroup/memory.max").read().strip())
    lines = [l for l in open(f"{a.run_dir}/phase2/bq/metadata_counts.csv") if l.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("n_rows"))
    bq = list(csv.DictReader(lines[start:]))[0]
    env = {
        "hw_gpu_model": rec(" / ".join(names), "model", "nvidia-smi", "nvidia-smi --query-gpu=name", "this box"),
        "hw_gpu_count": rec(len(smi), "GPUs", "nvidia-smi", "nvidia-smi --query-gpu=name"),
        "hw_gpu_vram_gb": rec(vram, "GB", "nvidia-smi", "nvidia-smi --query-gpu=memory.total"),
        "hw_host_ram_gb": rec(round(cg / 2**30, 1), "GiB", "/sys/fs/cgroup/memory.max", "cat /sys/fs/cgroup/memory.max", "container cgroup limit"),
        "sw_python": rec(platform.python_version(), "version", ".venv/bin/python", "python --version"),
        "sw_cupy": rec(cupy.__version__, "version", ".venv", "python -c 'import cupy'"),
        "sw_numpy": rec(numpy.__version__, "version", ".venv", "python -c 'import numpy'"),
        "sw_polars": rec(polars.__version__, "version", ".venv", "python -c 'import polars'"),
        "sw_cuda": rec(f"driver CUDA 13.2 (driver {driver}); nvcc {nvcc}", "version", "nvidia-smi / nvcc", "nvidia-smi; nvcc --version"),
        "sw_os": rec(os_name, "os", "/etc/os-release", "cat /etc/os-release"),
        "dataset_uniprot_release": rec("2026_01", "release", f"{a.run_dir}/phase2/data/stream_trembl_2026_01.log",
                                       "release fingerprint: old log 202,556,314 records == TrEMBL 2026_01 entry count (RESULTS.md U-005); data streamed from previous_releases/release-2026_01",
                                       "the paper states 2025_01; the data actually used (and used here) is 2026_01"),
        "dataset_access_date": rec("2026-09-02", "date", f"{a.run_dir}/phase2/data/stream_trembl_2026_01.log", "campaign download date"),
        "dataset_annotation_gb": rec(149.8, "GiB", f"{a.run_dir}/phase2/data/stream_trembl_2026_01.log",
                                     "tar member size of uniprot_trembl.dat.gz (2026_01) = 160,834,306,675 B", "= 160.8 GB decimal; the paper's '150 GB' is the GiB figure (aria2c convention in the old watcher log)"),
        "dataset_metadata_rows": rec(int(bq["n_rows"]), "rows", f"{a.run_dir}/phase2/bq/metadata_counts.csv", "bq query COUNT(*) on bigquery-public-data.deepmind_alphafold.metadata"),
        "dataset_metadata_accessions": rec(int(bq["n_accessions"]), "accessions", f"{a.run_dir}/phase2/bq/metadata_counts.csv", "COUNT(DISTINCT uniprotAccession)"),
        "dataset_plddt_pass_bq": rec(int(bq["n_plddt_ge50"]), "rows", f"{a.run_dir}/phase2/bq/metadata_counts.csv", "COUNTIF(globalMetricValue >= 50)"),
        "misc_row_split_gpus": rec(2, "GPUs", "PROGRESS.md", "campaign configuration", "all mining on 2×RTX 3090 (row-split) unless noted"),
    }
    json.dump(env, open(a.out, "w"), indent=1)
    print(f"env.json: {len(env)} keys")


if __name__ == "__main__":
    main()
