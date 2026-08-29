"""GCS upload + Polars sink_parquet helpers — consolidated module.

Three pathways:

1. **Polars sink_parquet -> gs://** (zero local disk).
   Pass `polars_storage_options()` as `storage_options` kwarg. Auth via ADC
   authorized_user JSON (refresh_token-backed, OAuth2 auto-refresh).

2. **Local file -> gs://** chunked parallel upload via `upload_to_uri()`.
   8-stream concurrent transfer beats single-stream `gcloud cp` 5-10x on
   large parquets.

3. **Background drain pattern** via `GCSUploader` for callers that produce
   files in a loop and want non-blocking uploads.

Credential lookup order:
  1. ET_MINER_GCS_CREDENTIALS env var (path)
  2. <repo_root>/dev-credentials.json
  3. ~/.config/gcloud/application_default_credentials.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from et_miner import _env

if TYPE_CHECKING:
    from google.cloud.storage import Client


CHUNK_SIZE_MB = 128
PARALLEL_WORKERS = 8
PARALLEL_THRESHOLD_MB = 64


def is_upload_enabled() -> bool:
    return _env.upload_gcs_enabled()


def is_gs_uri(uri: object) -> bool:
    return isinstance(uri, str) and uri.startswith("gs://")


def parse_gs_url(url: str) -> tuple[str, str]:
    if not is_gs_uri(url):
        raise ValueError(f"expected gs:// URL, got {url!r}")
    rest = url[5:]
    if "/" not in rest:
        return rest, ""
    bucket, prefix = rest.split("/", 1)
    return bucket, prefix


def join_gs_uri(base: str, *parts: str) -> str:
    """Join a gs:// or local path with extra components, no double-slash."""
    base = base.rstrip("/")
    suffix = "/".join(p.strip("/") for p in parts if p)
    return f"{base}/{suffix}" if suffix else base


_resolve_cache: dict[tuple[str, int], str] = {}


def clear_resolve_cache() -> None:
    """Invalidate resolve_k_parquet cache between runs or output_dir changes."""
    _resolve_cache.clear()


def resolve_k_parquet(base_dir: str, k: int) -> str:
    """Resolve path for K-level parquet: single file or partitioned directory.

    Partitioned output (parallel flush) writes to frequent_k{k}/part_*.parquet.
    Legacy output writes to frequent_k{k}.parquet.
    Results are cached per (base_dir, k) — call clear_resolve_cache() between runs.
    """
    cache_key = (base_dir, k)
    if cache_key in _resolve_cache:
        return _resolve_cache[cache_key]

    single = join_gs_uri(base_dir, f"frequent_k{k}.parquet")
    part_dir = join_gs_uri(base_dir, f"frequent_k{k}")

    if is_gs_uri(base_dir):
        try:
            import pyarrow.fs as pa_fs

            fs = pyarrow_gcs_filesystem()
            bucket, blob = parse_gs_url(part_dir)
            info = fs.get_file_info(f"{bucket}/{blob}")
            if info.type == pa_fs.FileType.Directory:
                _resolve_cache[cache_key] = part_dir
                return part_dir
        except Exception:  # noqa: BLE001
            pass
        _resolve_cache[cache_key] = single
        return single

    if os.path.isdir(part_dir):
        _resolve_cache[cache_key] = part_dir
        return part_dir
    _resolve_cache[cache_key] = single
    return single


def default_creds_path() -> Path | None:
    env = _env.gcs_credentials_path()
    if env:
        p = Path(env)
        return p if p.exists() else None
    candidates = [
        Path.cwd() / "dev-credentials.json",
        Path.home() / ".config/gcloud/application_default_credentials.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def polars_storage_options(creds_path: Path | None = None) -> dict[str, str]:
    """storage_options for `pl.scan_parquet` / `LazyFrame.sink_parquet` on gs://.

    object_store accepts authorized_user ADC via google_application_credentials,
    handling OAuth2 refresh internally.
    """
    creds_path = creds_path or default_creds_path()
    if creds_path is None:
        raise FileNotFoundError(
            "GCS credentials not found — set ET_MINER_GCS_CREDENTIALS or run `gcloud auth application-default login`",
        )
    return {"google_application_credentials": str(creds_path)}


def pyarrow_gcs_filesystem(creds_path: Path | None = None):
    """pyarrow.fs.GcsFileSystem authenticated via ADC authorized_user JSON.

    For chunked ParquetWriter direct-to-gs:// writes that bypass Polars'
    sink_parquet RAM-pressure (Polars materializes full Arrow table even
    in streaming mode for in-memory sources). PyArrow auto-discovers ADC
    via GOOGLE_APPLICATION_CREDENTIALS env var.
    """
    import pyarrow.fs as pa_fs

    creds_path = creds_path or default_creds_path()
    if creds_path is not None:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    return pa_fs.GcsFileSystem()


def make_client(creds_path: Path | None = None) -> Client:
    """google-cloud-storage Client. Tries ADC JSON first, falls back to GCS_TOKEN env."""
    from google.cloud.storage import Client as _Client
    from google.oauth2.credentials import Credentials

    creds_path = creds_path or default_creds_path()
    if creds_path is not None:
        data = json.loads(creds_path.read_text())
        creds = Credentials(
            token=None,
            refresh_token=data["refresh_token"],
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
            quota_project_id=data.get("quota_project_id"),
        )
        return _Client(credentials=creds, project=data.get("quota_project_id"))

    raw_token = _env.gcs_token()
    if raw_token:
        return _Client(credentials=Credentials(token=raw_token))

    raise FileNotFoundError(
        "GCS credentials not found — set ET_MINER_GCS_CREDENTIALS, place "
        "dev-credentials.json in the working directory, run "
        "`gcloud auth application-default login`, or set GCS_TOKEN env var",
    )


def upload_one(
    local: Path,
    bucket: str,
    blob_name: str,
    client: Client | None = None,
) -> tuple[float, float]:
    """Upload single file. Chunked parallel if > PARALLEL_THRESHOLD_MB.

    Returns (elapsed_seconds, MB_per_second).
    """
    from google.cloud.storage import transfer_manager

    client = client or make_client()
    size = local.stat().st_size
    blob = client.bucket(bucket).blob(blob_name)
    t0 = time.time()
    if size > PARALLEL_THRESHOLD_MB * 1024 * 1024:
        transfer_manager.upload_chunks_concurrently(
            str(local),
            blob,
            chunk_size=CHUNK_SIZE_MB * 1024 * 1024,
            max_workers=PARALLEL_WORKERS,
        )
    else:
        blob.upload_from_filename(str(local))
    dt = time.time() - t0
    mbs = size / dt / 1024**2 if dt > 0 else 0.0
    return dt, mbs


def upload_to_uri(
    local: Path,
    gs_uri: str,
    client: Client | None = None,
) -> tuple[float, float]:
    """Upload local file to gs://bucket/path. Trailing `/` keeps local filename."""
    bucket, blob_name = parse_gs_url(gs_uri)
    if not blob_name or blob_name.endswith("/"):
        blob_name = (blob_name.rstrip("/") + "/" + local.name).lstrip("/")
    return upload_one(local, bucket, blob_name, client)


def upload_file(local_path: str | Path, gcs_prefix: str) -> None:
    """Upload a single file to {ET_MINER_GCS_BUCKET}/{prefix}/{filename}.

    Uses chunked parallel transfer when google-cloud-storage is importable,
    falls back to `gcloud storage cp` subprocess otherwise.
    """
    local = Path(local_path)
    fname = local.name
    gs_uri = f"{_env.gcs_bucket()}/{gcs_prefix}/{fname}"

    try:
        from google.cloud import storage  # noqa: F401

        dt, mbs = upload_to_uri(local, gs_uri)
        logger.info(f"GCS upload {fname}: OK ({mbs:.0f} MB/s, {dt:.1f}s)")
        return
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning(f"GCS upload {fname}: chunked-parallel failed ({e}), falling back to gcloud cp")

    if not shutil.which("gcloud"):
        logger.warning(f"gcloud CLI not found — skipping upload of {fname}")
        return
    r = subprocess.run(
        ["gcloud", "storage", "cp", "-q", str(local), gs_uri],
        capture_output=True,
        timeout=1800,
    )
    if r.returncode == 0:
        logger.info(f"GCS upload {fname}: OK (gcloud cp)")
    else:
        stderr = r.stderr.decode().strip()[:200] if r.stderr else "unknown error"
        logger.warning(f"GCS upload {fname}: FAIL({r.returncode}) — {stderr}")


class GCSUploader:
    """Non-blocking GCS uploader with background workers + two-phase shutdown.

    Lifecycle:
      drain() — wait for pending futures, clear list. Executor stays alive.
      close() — shutdown the executor (idempotent).

    The split exists because callers (apriori K-loop) need to drain between
    K-levels without killing the executor mid-run.
    """

    def __init__(
        self,
        prefix: str,
        max_workers: int = 1,
        enabled: bool | None = None,
    ):
        self.enabled = is_upload_enabled() if enabled is None else enabled
        self.prefix = prefix
        self._executor = ThreadPoolExecutor(max_workers=max_workers) if self.enabled else None
        self._futures: list[Future] = []
        self._closed = False

    def upload(self, path: str | Path) -> None:
        if self._executor is None or self._closed:
            return
        self._futures.append(
            self._executor.submit(upload_file, str(path), self.prefix),
        )

    def upload_to_uri(self, local: str | Path, gs_uri: str) -> None:
        """Schedule upload of `local` to absolute `gs://bucket/path`."""
        if self._executor is None or self._closed:
            return
        self._futures.append(
            self._executor.submit(upload_to_uri, Path(local), gs_uri),
        )

    def drain(self, shutdown: bool = False) -> None:
        """Wait for pending futures + clear list. Optionally shutdown executor."""
        if self._futures:
            logger.info(f"Waiting for {len(self._futures)} GCS upload(s)...")
            for fut in self._futures:
                try:
                    fut.result(timeout=1800)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"GCS upload failed: {e}")
            self._futures.clear()
        if shutdown:
            self.close(wait=True)

    def close(self, wait: bool = True) -> None:
        """Shutdown the executor. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
