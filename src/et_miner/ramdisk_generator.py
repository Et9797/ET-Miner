"""
Ramdisk Data Generator for H200 GPU Pipeline Optimization

Manages tmpfs ramdisk for pre-generating and caching wave data during
GPU pipeline execution. Eliminates GPU idle time by pre-loading transaction
waves to RAM while previous wave is processing on GPU.

Wave Architecture:
- Each wave = 3.4B transactions (~130GB per GPU at 10 items/tx)
- Data format: CSR sparse matrix (indices + indptr arrays)
- Storage: Binary format for minimal disk footprint
- Persistence: Ramdisk tmpfs for speed, auto-cleanup on unmount

Usage:
    # Setup ramdisk (1400GB for 8x H200)
    ramdisk_path = setup_ramdisk(size_gb=1400)

    # Generate waves to ramdisk
    for wave_id in range(num_waves):
        generate_wave_to_disk(
            wave_id=wave_id,
            n_rows=3_400_000_000,  # 3.4B per wave
            path=ramdisk_path
        )

    # Load wave back to CuPy when ready for GPU
    indices_gpu, indptr_gpu = load_wave_from_disk(wave_id, ramdisk_path)

    # Cleanup when done
    cleanup_ramdisk(ramdisk_path)
"""

import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional

from loguru import logger

__all__ = [
    'setup_ramdisk',
    'cleanup_ramdisk',
    'get_ramdisk_info',
    'generate_wave_to_disk',
    'load_wave_from_disk',
    'delete_wave',
    'list_waves',
]


# =============================================================================
# Ramdisk Lifecycle Management
# =============================================================================

def setup_ramdisk(size_gb: int, mount_point: str = "/mnt/ramdisk") -> Path:
    """
    Setup tmpfs ramdisk for wave data caching.

    Creates a tmpfs mount at the specified location with given size.
    Ramdisk provides ~10x faster I/O than disk and uses zero persistent storage.

    Args:
        size_gb: Size of ramdisk in GB (e.g., 1400 for H200 server)
        mount_point: Where to mount (default: /mnt/ramdisk)

    Returns:
        Path to ramdisk mount point

    Raises:
        RuntimeError: If setup fails (permissions, disk space, etc.)
        PermissionError: If not running as root (required for mount)

    Example:
        >>> ramdisk = setup_ramdisk(size_gb=1400)
        >>> print(f"Ramdisk mounted at {ramdisk}")
        >>> # Files written to ramdisk_path are stored in RAM
    """
    path = Path(mount_point)

    # Check permissions
    if os.geteuid() != 0:
        raise PermissionError(
            f"setup_ramdisk() requires root privileges to mount tmpfs. "
            f"Run with: sudo python or use: echo {size_gb}G | mount -t tmpfs -o size=$(cat) tmpfs {mount_point}"
        )

    try:
        # Create mount point if needed
        if not path.exists():
            logger.info(f"[RAMDISK] Creating mount point: {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            # Check if already mounted
            result = subprocess.run(
                ['mountpoint', str(path)],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"[RAMDISK] Already mounted at {path}")
                # Verify size matches request
                stat_info = get_ramdisk_info(path)
                if stat_info and stat_info['size_gb'] >= size_gb:
                    logger.info(f"[RAMDISK] Size OK: {stat_info['size_gb']:.1f}GB")
                    return path
                else:
                    logger.warning(f"[RAMDISK] WARNING: Requested {size_gb}GB but mounted with {stat_info['size_gb']:.1f}GB")
                    return path

        # Mount tmpfs with size limit
        size_str = f"{size_gb}G"
        logger.info(f"[RAMDISK] Mounting {size_str} tmpfs at {path}")

        result = subprocess.run(
            ['mount', '-t', 'tmpfs', '-o', f'size={size_str}', 'tmpfs', str(path)],
            capture_output=True,
            timeout=30,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to mount ramdisk: {result.stderr}"
            )

        logger.info(f"[RAMDISK] Successfully mounted {size_str} at {path}")

        # Verify mount
        stat_info = get_ramdisk_info(path)
        if stat_info:
            logger.info(f"[RAMDISK] Verified: {stat_info['size_gb']:.1f}GB total, "
                        f"{stat_info['free_gb']:.1f}GB available")

        return path

    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout during ramdisk setup")
    except Exception as e:
        raise RuntimeError(f"Ramdisk setup failed: {e}")


def cleanup_ramdisk(path: Path) -> bool:
    """
    Unmount and cleanup ramdisk.

    Safely unmounts tmpfs and optionally removes mount point.
    Handles cases where ramdisk is already unmounted.

    Args:
        path: Path to ramdisk mount point

    Returns:
        True if cleanup successful, False if already unmounted

    Raises:
        RuntimeError: If unmount fails (files still in use, permission denied)
        PermissionError: If not running as root

    Example:
        >>> cleanup_ramdisk(Path("/mnt/ramdisk"))
        >>> # Ramdisk unmounted, data lost, mount point remains
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"[RAMDISK] Mount point does not exist: {path}")
        return False

    # Check permissions
    if os.geteuid() != 0:
        raise PermissionError(
            f"cleanup_ramdisk() requires root privileges. "
            f"Run with: sudo python or manually: umount {path}"
        )

    try:
        # Check if mounted
        result = subprocess.run(
            ['mountpoint', str(path)],
            capture_output=True,
            timeout=5
        )

        if result.returncode != 0:
            logger.warning(f"[RAMDISK] Not mounted: {path}")
            return False

        # Unmount
        logger.info(f"[RAMDISK] Unmounting {path}")
        result = subprocess.run(
            ['umount', str(path)],
            capture_output=True,
            timeout=30,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to unmount ramdisk at {path}: {result.stderr}"
            )

        logger.info(f"[RAMDISK] Successfully unmounted {path}")
        return True

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Timeout during ramdisk cleanup at {path}")
    except Exception as e:
        raise RuntimeError(f"Ramdisk cleanup failed: {e}")


def get_ramdisk_info(path: Path) -> Optional[dict]:
    """
    Get ramdisk usage statistics.

    Args:
        path: Path to ramdisk mount point

    Returns:
        Dict with keys: size_gb, used_gb, free_gb, percent_used
        Returns None if path is not a mounted tmpfs

    Example:
        >>> info = get_ramdisk_info(Path("/mnt/ramdisk"))
        >>> print(f"Used: {info['used_gb']:.1f}GB / {info['size_gb']:.1f}GB")
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        result = subprocess.run(
            ['df', str(path)],
            capture_output=True,
            timeout=5,
            text=True
        )

        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None

        # Parse df output: Filesystem 1K-blocks Used Available Use% Mounted
        parts = lines[1].split()
        if len(parts) < 4:
            return None

        total_kb = int(parts[1])
        used_kb = int(parts[2])
        free_kb = int(parts[3])

        total_gb = total_kb / (1024 * 1024)
        used_gb = used_kb / (1024 * 1024)
        free_gb = free_kb / (1024 * 1024)
        percent = (used_gb / total_gb * 100) if total_gb > 0 else 0

        return {
            'size_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'percent_used': percent,
        }

    except (subprocess.TimeoutExpired, ValueError, IndexError):
        return None


# =============================================================================
# Wave Generation
# =============================================================================

def generate_wave_to_disk(
    wave_id: int,
    n_rows: int,
    path: Path,
    n_cols: int = 1000,
    avg_items: int = 10,
    seed: int = 42,
    dtype_indices: str = 'int64',
    dtype_indptr: str = 'int64',
) -> Tuple[int, int]:
    """
    Generate random transaction wave and save to ramdisk as binary CSR format.

    Generates a sparse matrix wave (CSR format) with realistic transaction
    distributions. Data is written as two binary files:
    - {wave_id}.indices: Column indices
    - {wave_id}.indptr: Row pointers

    This is a CPU-based generator. For GPU-based generation, use
    cuda_csr_build.generate_csr_gpu() and then save results.

    Args:
        wave_id: Wave identifier (0, 1, 2, ...)
        n_rows: Number of transactions in wave
        path: Ramdisk mount point path
        n_cols: Number of items (columns)
        avg_items: Average items per transaction
        seed: Random seed for reproducibility
        dtype_indices: numpy dtype for indices (int32 or int64)
        dtype_indptr: numpy dtype for indptr (int32 or int64)

    Returns:
        Tuple of (n_rows, nnz) for validation

    Raises:
        OSError: If write fails (disk full, permission denied)
        ValueError: If parameters are invalid
        ImportError: If numpy not available

    Example:
        >>> n_rows, nnz = generate_wave_to_disk(
        ...     wave_id=0,
        ...     n_rows=3_400_000_000,
        ...     path=Path("/mnt/ramdisk")
        ... )
        >>> print(f"Wave 0: {n_rows:,} rows, {nnz:,} non-zeros")
    """
    import numpy as np

    path = Path(path)

    # Validate parameters
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if n_cols <= 0:
        raise ValueError(f"n_cols must be positive, got {n_cols}")
    if avg_items <= 0 or avg_items > n_cols:
        raise ValueError(f"avg_items must be in (0, {n_cols}], got {avg_items}")

    # Check disk space
    stat_info = get_ramdisk_info(path)
    if stat_info is None:
        raise OSError(f"Cannot access ramdisk at {path}")

    # Estimate required space (rough)
    # indptr: (n_rows + 1) * 8 bytes
    # indices: ~n_rows * avg_items * 8 bytes
    nnz_estimate = int(n_rows * avg_items)
    indptr_size = (n_rows + 1) * 8
    indices_size = nnz_estimate * 8
    total_size = (indptr_size + indices_size) / (1024**3)

    if stat_info['free_gb'] < total_size * 1.1:  # 10% safety margin
        raise OSError(
            f"Insufficient ramdisk space: need {total_size:.1f}GB but only "
            f"{stat_info['free_gb']:.1f}GB available"
        )

    try:
        # Generate items per row (Poisson-like)
        logger.info(f"[WAVE {wave_id}] Generating {n_rows:,} transactions...")
        t0 = time.time()

        rng = np.random.RandomState(seed)
        items_per_row = np.clip(
            rng.poisson(avg_items, size=n_rows).astype(np.int64),
            1,
            n_cols
        )

        # Build indptr
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(items_per_row, out=indptr[1:])
        nnz = int(indptr[-1])

        # Generate random indices
        indices = rng.randint(0, n_cols, size=nnz, dtype=np.int64)

        gen_time = time.time() - t0
        logger.info(f"[WAVE {wave_id}] Generated in {gen_time:.1f}s: "
                    f"{nnz:,} non-zeros ({n_rows/gen_time/1e6:.1f}M txn/s)")

        # Write to binary files
        logger.info(f"[WAVE {wave_id}] Writing to {path}...")
        t0 = time.time()

        indices_file = path / f"{wave_id}.indices.bin"
        indptr_file = path / f"{wave_id}.indptr.bin"

        # Write indices
        with open(indices_file, 'wb') as f:
            # Header: magic(4) + n_rows(8) + nnz(8) + dtype_code(1) = 21 bytes
            header = struct.pack('<4sQQB',
                b'CSR!',
                np.uint64(n_rows),
                np.uint64(nnz),
                1 if dtype_indices == 'int64' else 0
            )
            f.write(header)
            indices.astype(np.dtype(dtype_indices)).tofile(f)

        # Write indptr
        with open(indptr_file, 'wb') as f:
            # Header: magic(4) + n_rows(8) + nnz(8) + dtype_code(1) = 21 bytes
            header = struct.pack('<4sQQB',
                b'CSR!',
                np.uint64(n_rows),
                np.uint64(nnz),
                1 if dtype_indptr == 'int64' else 0
            )
            f.write(header)
            indptr.astype(np.dtype(dtype_indptr)).tofile(f)

        write_time = time.time() - t0
        indices_size = indices_file.stat().st_size / (1024**3)
        indptr_size = indptr_file.stat().st_size / (1024**3)

        logger.info(f"[WAVE {wave_id}] Wrote in {write_time:.1f}s: "
                    f"indices={indices_size:.2f}GB, indptr={indptr_size:.2f}GB")

        # Update ramdisk stats
        stat_info = get_ramdisk_info(path)
        if stat_info:
            logger.info(f"[WAVE {wave_id}] Ramdisk: {stat_info['used_gb']:.1f}GB / "
                        f"{stat_info['size_gb']:.1f}GB ({stat_info['percent_used']:.0f}%)")

        return n_rows, nnz

    except OSError as e:
        raise OSError(f"Failed to write wave {wave_id}: {e}")
    except Exception as e:
        raise RuntimeError(f"Wave generation failed: {e}")


# =============================================================================
# Wave Loading
# =============================================================================

def load_wave_from_disk(
    wave_id: int,
    path: Path,
    dtype_indices: str = 'int64',
    dtype_indptr: str = 'int64',
) -> Tuple:
    """
    Load wave CSR data from ramdisk into CuPy arrays on GPU.

    Reads binary CSR files from ramdisk and transfers to GPU memory
    (via CuPy). Returns arrays in format ready for GPU bitvec construction.

    Args:
        wave_id: Wave identifier
        path: Ramdisk mount point path
        dtype_indices: Expected dtype for indices
        dtype_indptr: Expected dtype for indptr

    Returns:
        Tuple of (indices_gpu, indptr_gpu) as CuPy arrays

    Raises:
        FileNotFoundError: If wave files not found
        IOError: If read fails or data corrupted
        ImportError: If CuPy not available

    Example:
        >>> indices_gpu, indptr_gpu = load_wave_from_disk(0, Path("/mnt/ramdisk"))
        >>> # Data is now in GPU memory, ready for bitvec conversion
        >>> bitvecs = csr_to_bitvecs_gpu(indptr_gpu, indices_gpu, ...)
    """
    import numpy as np

    path = Path(path)

    indices_file = path / f"{wave_id}.indices.bin"
    indptr_file = path / f"{wave_id}.indptr.bin"

    # Check files exist
    if not indices_file.exists():
        raise FileNotFoundError(f"Wave {wave_id} indices file not found: {indices_file}")
    if not indptr_file.exists():
        raise FileNotFoundError(f"Wave {wave_id} indptr file not found: {indptr_file}")

    try:
        logger.info(f"[WAVE {wave_id}] Loading from {path}...")
        t0 = time.time()

        # Read indices
        with open(indices_file, 'rb') as f:
            header = f.read(21)  # magic(4) + n_rows(8) + nnz(8) + dtype(1)
            if header[:4] != b'CSR!':
                raise IOError(f"Invalid indices header magic: {header[:4]}")
            n_rows, nnz = struct.unpack('<QQ', header[4:20])
            indices = np.fromfile(f, dtype=np.dtype(dtype_indices), count=nnz)

        # Read indptr
        with open(indptr_file, 'rb') as f:
            header = f.read(21)
            if header[:4] != b'CSR!':
                raise IOError(f"Invalid indptr header magic: {header[:4]}")
            indptr = np.fromfile(f, dtype=np.dtype(dtype_indptr), count=n_rows + 1)

        # Verify data integrity
        if len(indices) != nnz:
            raise IOError(f"Expected {nnz} indices, got {len(indices)}")
        if len(indptr) != n_rows + 1:
            raise IOError(f"Expected {n_rows + 1} indptr, got {len(indptr)}")
        if int(indptr[-1]) != nnz:
            raise IOError(f"indptr[-1]={indptr[-1]} != nnz={nnz}")

        read_time = time.time() - t0
        logger.info(f"[WAVE {wave_id}] Read in {read_time:.1f}s: "
                    f"{n_rows:,} rows, {nnz:,} non-zeros")

        # Transfer to GPU
        try:
            import cupy as cp

            logger.info(f"[WAVE {wave_id}] Transferring to GPU...")
            t0 = time.time()

            indices_gpu = cp.asarray(indices)
            indptr_gpu = cp.asarray(indptr)

            xfer_time = time.time() - t0
            indices_size = (len(indices) * indices.itemsize) / (1024**3)
            indptr_size = (len(indptr) * indptr.itemsize) / (1024**3)

            logger.info(f"[WAVE {wave_id}] GPU transfer in {xfer_time:.1f}s: "
                        f"indices={indices_size:.2f}GB, indptr={indptr_size:.2f}GB")

            return indices_gpu, indptr_gpu

        except ImportError:
            raise ImportError(
                "CuPy not available. Install with: pip install cupy-cuda12x"
            )

    except FileNotFoundError:
        raise
    except (IOError, struct.error) as e:
        raise IOError(f"Failed to read wave {wave_id}: {e}")
    except Exception as e:
        raise RuntimeError(f"Wave loading failed: {e}")


# =============================================================================
# Utilities
# =============================================================================

def delete_wave(wave_id: int, path: Path) -> bool:
    """
    Delete a wave from ramdisk to free space.

    Args:
        wave_id: Wave identifier
        path: Ramdisk mount point path

    Returns:
        True if deleted, False if files not found
    """
    path = Path(path)
    indices_file = path / f"{wave_id}.indices.bin"
    indptr_file = path / f"{wave_id}.indptr.bin"

    deleted = False

    for f in [indices_file, indptr_file]:
        if f.exists():
            try:
                f.unlink()
                deleted = True
            except OSError as e:
                logger.error(f"[ERROR] Failed to delete {f}: {e}")

    if deleted:
        stat_info = get_ramdisk_info(path)
        if stat_info:
            logger.info(f"[WAVE {wave_id}] Deleted. Ramdisk: "
                        f"{stat_info['free_gb']:.1f}GB free")

    return deleted


def list_waves(path: Path) -> list:
    """
    List all waves present in ramdisk.

    Args:
        path: Ramdisk mount point path

    Returns:
        List of wave IDs found
    """
    path = Path(path)
    waves = set()

    if not path.exists():
        return []

    for f in path.glob("*.indices.bin"):
        try:
            wave_id = int(f.stem.split('.')[0])
            waves.add(wave_id)
        except ValueError:
            pass

    return sorted(list(waves))


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == '__main__':
    import tempfile
    import shutil

    logger.info("Testing ramdisk generator (requires root)...")

    # For testing, use temp directory instead of real ramdisk
    test_dir = Path(tempfile.mkdtemp(prefix="ramdisk_test_"))
    logger.info(f"Using test directory: {test_dir}")

    try:
        # Test wave generation
        logger.info("[TEST] Generating test wave...")
        n_rows, nnz = generate_wave_to_disk(
            wave_id=0,
            n_rows=100_000,  # Small test
            path=test_dir,
            n_cols=1000,
            avg_items=10,
        )
        logger.info(f"OK: Generated {n_rows:,} rows, {nnz:,} non-zeros")

        # Test wave loading (without GPU)
        logger.info("[TEST] Loading test wave (numpy only)...")
        import numpy as np
        with open(test_dir / "0.indices.bin", 'rb') as f:
            header = f.read(17)
            data = np.fromfile(f, dtype=np.int64)
            logger.info(f"OK: Loaded {len(data):,} indices")

        # Test listing
        logger.info("[TEST] Listing waves...")
        waves = list_waves(test_dir)
        logger.info(f"OK: Found waves: {waves}")

        # Test deletion
        logger.info("[TEST] Deleting wave...")
        delete_wave(0, test_dir)
        waves = list_waves(test_dir)
        logger.info(f"OK: Remaining waves: {waves}")

    finally:
        # Cleanup
        logger.info(f"[TEST] Cleaning up {test_dir}...")
        shutil.rmtree(test_dir, ignore_errors=True)

    logger.info("All tests passed!")
