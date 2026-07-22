"""Download the demo and benchmark data into data/.

- Chinook (demo db): fetched directly from the official GitHub release.
- Spider (benchmark): hosted on Google Drive; we try gdown, and if that fails
  print manual instructions. If data/spider_data.zip already exists (manual
  download), it is just extracted.
"""

import shutil
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

CHINOOK_URL = (
    "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
)
# File id of spider_data.zip from the official Spider page (yale-lily.github.io/spider).
SPIDER_GDRIVE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
SPIDER_ZIP = DATA / "spider_data.zip"
SPIDER_DIR = DATA / "spider"


def download_chinook() -> None:
    dest = DATA / "chinook.db"
    if dest.exists():
        print(f"already present: {dest}")
        return
    print(f"downloading Chinook -> {dest}")
    resp = requests.get(CHINOOK_URL, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"done ({dest.stat().st_size / 1e6:.1f} MB)")


def download_spider() -> None:
    if (SPIDER_DIR / "dev.json").exists():
        print(f"already present: {SPIDER_DIR}")
        return

    if not SPIDER_ZIP.exists():
        try:
            import gdown

            print("downloading Spider from Google Drive (~1 GB)…")
            gdown.download(id=SPIDER_GDRIVE_ID, output=str(SPIDER_ZIP), quiet=False)
        except Exception as e:
            print(f"\nAutomatic Spider download failed: {e}", file=sys.stderr)
            print(
                "\nManual fallback:\n"
                "  1. Open https://yale-lily.github.io/spider and download spider_data.zip\n"
                f"  2. Save it as {SPIDER_ZIP}\n"
                "  3. Re-run this script (it will extract it).",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"extracting {SPIDER_ZIP}…")
    extract_dir = DATA / "_spider_extract"
    with zipfile.ZipFile(SPIDER_ZIP) as zf:
        zf.extractall(extract_dir)

    # the zip's top-level folder name has varied over releases; locate dev.json + database/
    candidates = [p.parent for p in extract_dir.rglob("dev.json") if (p.parent / "database").is_dir()]
    if not candidates:
        print(f"could not find dev.json + database/ inside {extract_dir}", file=sys.stderr)
        sys.exit(1)
    shutil.move(str(candidates[0]), str(SPIDER_DIR))
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"done: {SPIDER_DIR}")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    download_chinook()
    download_spider()
