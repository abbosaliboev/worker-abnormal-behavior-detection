"""
Downloads KTH Action Dataset (Running + Walking classes).

KTH Action Dataset — Schuldt et al., ICPR 2004
  URL: https://www.csc.kth.se/cvap/actions/
  Size: ~290 MB (running + walking)
  Subjects: 25, FPS: 25

Usage:
    python -m datasets.download_running
    python -m datasets.download_running --output_dir data/running_dataset
"""

import os
import ssl
import urllib.request
import zipfile
import argparse

ssl._create_default_https_context = ssl._create_unverified_context

DEFAULT_OUTPUT = r"f:\Project_F\Company_Abnormal_Project\data\running_dataset"

KTH_URLS = {
    "running": "https://www.csc.kth.se/cvap/actions/running.zip",
    "walking": "https://www.csc.kth.se/cvap/actions/walking.zip",
}


def download_file(url: str, dest: str) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    def _progress(block, bsize, total):
        done = block * bsize
        if total > 0:
            print(f"\r  {min(100, done*100//total):3d}%  ({done/1e6:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        print()
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    print("=" * 55)
    print("  KTH Action Dataset — Download (Running + Walking)")
    print("=" * 55)
    print(f"  Output: {out}\n")

    for action, url in KTH_URLS.items():
        cls_dir  = os.path.join(out, action.capitalize())
        zip_path = os.path.join(out, f"{action}.zip")

        if os.path.isdir(cls_dir):
            n = len([f for f in os.listdir(cls_dir) if f.endswith(".avi")])
            if n > 0:
                print(f"  {action}: {n} videos already present, skipping.")
                continue

        print(f"  Downloading {action} from:\n    {url}")
        if not download_file(url, zip_path):
            print(f"  ERROR: Could not download {action}")
            continue

        print(f"  Extracting {action}.zip ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out)
        os.remove(zip_path)

        # KTH extracts to scenario subfolders — merge into one folder
        os.makedirs(cls_dir, exist_ok=True)
        for item in os.listdir(out):
            item_path = os.path.join(out, item)
            if item.startswith(action) and os.path.isdir(item_path):
                for f in os.listdir(item_path):
                    if f.endswith(".avi"):
                        os.rename(os.path.join(item_path, f),
                                  os.path.join(cls_dir, f))
                os.rmdir(item_path)

        n = len([f for f in os.listdir(cls_dir) if f.endswith(".avi")])
        print(f"  Done: {n} videos saved to {cls_dir}")

    print("\nDownload complete. Run evaluation:")
    print("  python -m evaluation.eval_running")


if __name__ == "__main__":
    main()
