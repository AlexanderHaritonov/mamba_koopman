"""Download what the project needs from KITTI raw, for every drive that has tracklets:
tracklets, calib, OXTS and image_02 frames. Only these files are pulled out of the
multi-GB *_sync.zip archives (remotezip, HTTP range requests). Reruns resume.

Same layout as kitty_data/, but with the full drive name as folder (drive numbers repeat across dates):

    raw_data/2011_09_26_drive_0001/
        calib/calib_{cam_to_cam,imu_to_velo,velo_to_cam}.txt
        tracklet_labels.xml
        2011_09_26_drive_0001_sync/
            image_02/{data/*.png, timestamps.txt}
            oxts/{data/*.txt, dataformat.txt, timestamps.txt}
"""
import argparse
import io
import os
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from remotezip import RemoteZip

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
RAW_URL = KITTI_URL + "raw_data/"
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data")


def list_drives():
    """All raw drives, as listed in KITTI's own raw_data_downloader.sh."""
    response = requests.get(KITTI_URL + "raw_data_downloader.zip", timeout=30)
    response.raise_for_status()
    script = zipfile.ZipFile(io.BytesIO(response.content)).read("raw_data_downloader.sh").decode()
    return list(dict.fromkeys(re.findall(r"\d{4}_\d{2}_\d{2}_drive_\d{4}", script)))


def has_tracklets(drive):
    return requests.head(f"{RAW_URL}{drive}/{drive}_tracklets.zip", timeout=30).status_code == 200


def target_path(name, folders):
    """Where an archive member goes inside the drive folder, or None if it isn't needed.

    Members look like 2011_09_26/calib_cam_to_cam.txt, 2011_09_26/<drive>_sync/tracklet_labels.xml
    and 2011_09_26/<drive>_sync/oxts/data/0000000000.txt.
    """
    _, _, rel = name.partition("/")  # drop the date folder
    if rel.startswith("calib_"):
        return os.path.join("calib", rel)
    _, _, inner = rel.partition("/")
    if inner == "tracklet_labels.xml":
        return inner
    if inner.startswith(folders) and not inner.endswith("/"):
        return rel
    return None


def fetch_drive(drive, out_dir, with_images, dry_run):
    """Fetch one drive's missing files. Returns (files, bytes) fetched, or to be fetched in a dry run."""
    print(f"{drive}: {'checking' if dry_run else 'downloading'}...", flush=True)
    folders = ("oxts/", "image_02/") if with_images else ("oxts/",)
    urls = [f"{RAW_URL}{drive[:10]}_calib.zip",
            f"{RAW_URL}{drive}/{drive}_tracklets.zip",
            f"{RAW_URL}{drive}/{drive}_sync.zip"]
    files = size = 0
    for url in urls:
        with RemoteZip(url) as archive:
            for info in archive.infolist():
                rel = target_path(info.filename, folders)
                if rel is None:
                    continue
                path = os.path.join(out_dir, drive, rel)
                if os.path.exists(path) and os.path.getsize(path) == info.file_size:
                    continue
                files += 1
                size += info.file_size
                if not dry_run:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "wb") as f:
                        f.write(archive.read(info))
    return files, size


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=RAW_DIR, help="target folder (default: raw_data next to this script)")
    parser.add_argument("--no-images", action="store_true", help="skip image_02: tracklets, calib and OXTS only")
    parser.add_argument("--drives", nargs="+", help="only these drives, e.g. 2011_09_26_drive_0001")
    parser.add_argument("--workers", type=int, default=4, help="drives fetched in parallel")
    parser.add_argument("--dry-run", action="store_true", help="only report what would be downloaded")
    args = parser.parse_args()

    drives = args.drives or list_drives()
    with ThreadPoolExecutor(args.workers) as pool:
        drives = [d for d, ok in zip(drives, pool.map(has_tracklets, drives)) if ok]
    print(f"{len(drives)} drives with tracklets")

    total_files = total_size = 0
    failed = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_drive, d, args.out, not args.no_images, args.dry_run): d for d in drives}
        for done, future in enumerate(as_completed(futures), 1):
            drive = futures[future]
            left = len(drives) - done
            try:
                files, size = future.result()
            except Exception as e:
                failed.append(drive)
                print(f"{drive}: failed ({e}), {left} drives left")
                continue
            total_files += files
            total_size += size
            print(f"{drive}: {files} files, {size / 1e6:.0f} MB, {left} drives left")

    print(f"{'to fetch' if args.dry_run else 'fetched'}: {total_files} files, {total_size / 1e9:.2f} GB")
    if failed:
        print("failed, rerun to resume:", " ".join(sorted(failed)))


if __name__ == "__main__":
    main()
