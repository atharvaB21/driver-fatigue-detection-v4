#!/usr/bin/env python3
"""
Download and extract the dlib 68-point facial-landmark predictor model.

Usage
-----
Run as a standalone script::

    python download_model.py

Or import and call programmatically::

    from download_model import download_model, check_model

    if not check_model():
        download_model()
"""

from __future__ import annotations

import bz2
import os
import sys
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
_MODEL_URL: str = (
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
)
_MODEL_FILENAME: str = "shape_predictor_68_face_landmarks.dat"
_MODELS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH: str = os.path.join(_MODELS_DIR, _MODEL_FILENAME)
_BZ2_PATH: str = _MODEL_PATH + ".bz2"


# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------


def check_model(model_path: Optional[str] = None) -> bool:
    """Return ``True`` if the landmark-predictor model file already exists.

    Parameters
    ----------
    model_path:
        Override path to check.  Defaults to ``models/<model_filename>``
        relative to this script.
    """
    path = model_path or _MODEL_PATH
    return os.path.isfile(path)


def download_model(
    url: str = _MODEL_URL,
    dest_dir: str = _MODELS_DIR,
    force: bool = False,
) -> str:
    """Download and extract the dlib shape-predictor model.

    Parameters
    ----------
    url:
        URL of the ``.bz2``-compressed model archive.
    dest_dir:
        Directory where the extracted ``.dat`` file will be saved.
    force:
        If ``True``, re-download even when the file already exists.

    Returns
    -------
    str
        Absolute path to the extracted model file.
    """
    model_path = os.path.join(dest_dir, _MODEL_FILENAME)
    bz2_path = model_path + ".bz2"

    # Skip if already present (unless forced) --------------------------------
    if not force and os.path.isfile(model_path):
        print(f"[OK] Model already exists at {model_path}")
        return model_path

    # Ensure target directory exists -----------------------------------------
    os.makedirs(dest_dir, exist_ok=True)

    # Download with progress -------------------------------------------------
    print(f"[DL] Downloading model from:\n    {url}")
    _download_with_progress(url, bz2_path)

    # Extract .bz2 -----------------------------------------------------------
    print("[..] Extracting bz2 archive ...")
    _extract_bz2(bz2_path, model_path)

    # Clean up compressed file -----------------------------------------------
    try:
        os.remove(bz2_path)
    except OSError:
        pass

    print(f"[OK] Model saved to {model_path}")
    return model_path


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------


def _reporthook(
    block_num: int, block_size: int, total_size: int
) -> None:
    """Console progress-bar callback for :func:`urllib.request.urlretrieve`.

    Parameters
    ----------
    block_num:
        Number of blocks transferred so far.
    block_size:
        Size of each block in bytes.
    total_size:
        Total size of the file (``-1`` if unknown).
    """
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded / total_size * 100.0)
        bar_length = 40
        filled = int(bar_length * percent / 100.0)
        bar = "#" * filled + "-" * (bar_length - filled)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\r    [{bar}] {percent:5.1f}%  ({mb_down:.1f}/{mb_total:.1f} MB)"
        )
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r    {mb_down:.1f} MB downloaded")
    sys.stdout.flush()


def _download_with_progress(url: str, dest_path: str) -> None:
    """Download *url* to *dest_path*, printing a progress bar to stdout.

    Parameters
    ----------
    url:
        Remote URL to fetch.
    dest_path:
        Local file path where the download is saved.

    Raises
    ------
    urllib.error.URLError
        If the download fails.
    """
    urllib.request.urlretrieve(url, dest_path, reporthook=_reporthook)
    print()  # newline after progress bar


def _extract_bz2(src_path: str, dest_path: str) -> None:
    """Extract a bz2-compressed file to *dest_path*.

    Reads in 64 KiB chunks to avoid loading the entire archive into memory.

    Parameters
    ----------
    src_path:
        Path to the ``.bz2`` file.
    dest_path:
        Destination path for the extracted content.

    Raises
    ------
    FileNotFoundError
        If *src_path* does not exist.
    """
    chunk_size = 65_536  # 64 KiB
    with bz2.open(src_path, "rb") as src, open(dest_path, "wb") as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)


# ---------------------------------------------------------------------------
#  CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_model()
