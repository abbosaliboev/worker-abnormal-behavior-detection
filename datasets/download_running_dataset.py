"""
Downloads Running and Walking clips for running detector evaluation.

Uses KTH Action Dataset (Schuldt et al., ICPR 2004):
  - 25 subjects, 4 scenarios (outdoor s1-s3, indoor s4)
  - Running and walking classes, 25 fps, 160x120
  - Total size: ~290 MB (running + walking)

Reference:
  C. Schuldt, I. Laptev, B. Caputo, "Recognizing Human Actions:
  A Local SVM Approach", ICPR 2004.
"""

import os
import sys
import ssl
import urllib.request
import zipfile
import argparse
from pathlib import Path

# Disable SSL verification (common issue on Windows with self-signed certs)
ssl._create_default_https_context = ssl._create_unverified_context

OUTPUT_DIR = r"f:\Project_F\Company_Abnormal_Project\data\running_dataset"

# KTH official download URLs
KTH_URLS = {
    "running": "https://www.csc.kth.se/cvap/actions/running.zip",
    "walking": "https://www.csc.kth.se/cvap/actions/walking.zip",
    "jogging": "https://www.csc.kth.se/cvap/actions/jogging.zip",
}

# Backup mirrors
BACKUP_URLS = {
    "running": "http://www.nada.kth.se/cvap/actions/running.zip",
    "walking": "http://www.nada.kth.se/cvap/actions/walking.zip",
}


def download_file(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _progress(block, bsize, total):
        done = block * bsize
        if total > 0:
            pct = min(100, done * 100 / total)
            print(f"\r  {pct:5.1f}%  ({done/1e6:.1f} / {total/1e6:.1f} MB)",
                  end="", flush=True)
        else:
            print(f"\r  {done/1e6:.1f} MB downloaded", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        return False


def download_and_extract(name: str, output_dir: str) -> bool:
    cls_dir = os.path.join(output_dir, name.capitalize())
    if os.path.isdir(cls_dir):
        vids = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(".avi")]
        if vids:
            print(f"  {name}: {len(vids)} videos already present, skipping.")
            return True

    zip_path = os.path.join(output_dir, f"{name}.zip")

    # Try primary URL, then backup
    urls = [KTH_URLS.get(name), BACKUP_URLS.get(name)]
    downloaded = False
    for url in urls:
        if url is None:
            continue
        print(f"  Downloading {name} from:\n    {url}")
        if download_file(url, zip_path):
            downloaded = True
            break

    if not downloaded:
        return False

    print(f"  Extracting {name}.zip ...")
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        # KTH extracts to a folder named e.g. "running_s1", "running_s2"...
        # Merge all scenario folders into one class folder
        os.makedirs(cls_dir, exist_ok=True)
        for item in os.listdir(output_dir):
            if item.startswith(name) and os.path.isdir(os.path.join(output_dir, item)):
                src = os.path.join(output_dir, item)
                for f in os.listdir(src):
                    if f.endswith(".avi"):
                        os.rename(os.path.join(src, f),
                                  os.path.join(cls_dir, f))
                os.rmdir(src)
        print(f"  Done. Videos saved to: {cls_dir}")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 60)
    print("  KTH Action Dataset — Running & Walking Download")
    print("=" * 60)
    print(f"  Output: {out}")
    print(f"  Expected size: ~290 MB")
    print()

    ok_run  = download_and_extract("running", out)
    ok_walk = download_and_extract("walking", out)

    print()
    if ok_run and ok_walk:
        run_dir  = os.path.join(out, "Running")
        walk_dir = os.path.join(out, "Walking")
        n_run  = len([f for f in os.listdir(run_dir)  if f.endswith(".avi")])
        n_walk = len([f for f in os.listdir(walk_dir) if f.endswith(".avi")])
        print(f"Download complete!")
        print(f"  Running : {n_run} clips  -> {run_dir}")
        print(f"  Walking : {n_walk} clips -> {walk_dir}")
        print()
        print("Next step:")
        print("  python -m evaluation.eval_running_kth")
    else:
        print("Download failed. Manual download instructions:")
        print()
        print("  1. Go to: https://www.csc.kth.se/cvap/actions/")
        print("  2. Download: running.zip  and  walking.zip")
        print(f"  3. Extract to: {out}")
        print(f"     {out}\\Running\\*.avi")
        print(f"     {out}\\Walking\\*.avi")
        print()
        print("  Then run: python -m evaluation.eval_running_kth")


if __name__ == "__main__":
    main()
