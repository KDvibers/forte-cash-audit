"""Download pinned official assets, verify SHA256, and unpack the local runtime.

This setup utility uses the internet. The analysis tools use only localhost.
"""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCE = "huggingface"
MODEL_LOCK_PATH = ROOT / "analyzer/runtime-lock.json"
DOWNLOAD_WORKERS = 4


def modelscope_url(asset):
    repository = asset["url"].split("/resolve/", 1)[0].removeprefix("https://huggingface.co/")
    return f"https://www.modelscope.cn/api/v1/models/{repository}/repo?Revision=master&FilePath={asset['name']}"


def download_model_ranges(asset, target):
    """Resume separate byte ranges, then verify the SHA256 of the complete file."""
    partial = target.with_name(target.name + ".partial")
    prefix = partial.stat().st_size if partial.exists() else 0
    total = asset["size"]
    if prefix > total:
        raise RuntimeError("Partial file is larger than the expected model")
    pieces = target.parent / (asset["name"] + ".parts")
    pieces.mkdir(exist_ok=True)
    block = 16 * 1024 * 1024
    ranges = [(start, min(start + block, total) - 1) for start in range(prefix, total, block)]
    url = asset["url"] if MODEL_SOURCE == "huggingface-range" else modelscope_url(asset)

    def fetch(span):
        start, end = span
        piece = pieces / f"{start}-{end}.bin"
        expected = end - start + 1
        if piece.exists() and piece.stat().st_size == expected:
            return piece
        for attempt in range(4):
            try:
                have = piece.stat().st_size if piece.exists() else 0
                if have > expected:
                    raise RuntimeError("Cached byte range is larger than expected")
                if have == expected:
                    return piece
                tail = piece.with_suffix(".tail")
                request_start = start + have
                transfer = subprocess.run(
                    ["curl.exe", "--location", "--fail", "--silent", "--show-error",
                     "--connect-timeout", "20", "--max-time", "600",
                     "--range", f"{request_start}-{end}", "--output", str(tail),
                     "--write-out", "%{http_code}|%header{content-range}", url],
                    capture_output=True, text=True, timeout=610,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if transfer.stdout.strip() != f"206|bytes {request_start}-{end}/{total}":
                    if transfer.returncode:
                        raise OSError(f"curl transfer failed with code {transfer.returncode}")
                    raise RuntimeError("Download server did not honor the requested byte range")
                if tail.stat().st_size > expected - have:
                    raise RuntimeError("Response body exceeds requested byte range")
                with piece.open("ab") as output, tail.open("rb") as source:
                    while chunk := source.read(4 * 1024 * 1024):
                        output.write(chunk)
                if piece.stat().st_size != expected:
                    raise OSError("Incomplete model byte range")
                print(f"{asset['name']}: block {start // block + 1} ready", flush=True)
                return piece
            except RuntimeError:
                raise
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        completed = list(pool.map(fetch, ranges))
    with partial.open("ab") as output:
        for piece in completed:
            with piece.open("rb") as source:
                while chunk := source.read(4 * 1024 * 1024):
                    output.write(chunk)
    if partial.stat().st_size != total or digest(partial) != asset["sha256"]:
        raise RuntimeError(f"SHA256 mismatch: {asset['name']}")
    partial.replace(target)
    print(f"Verified: {asset['name']}", flush=True)
    return target


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(asset):
    target = ROOT / asset["target"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) == asset["sha256"]:
        print(f"Already verified: {asset['name']}", flush=True)
        return target
    is_model = asset["url"].startswith("https://huggingface.co/")
    if is_model and MODEL_SOURCE in {"modelscope", "huggingface-range"}:
        return download_model_ranges(asset, target)
    if is_model and MODEL_SOURCE == "huggingface":
        # The official chunked downloader avoids slow single-stream CDN transfers.
        os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
        os.environ.setdefault("HF_XET_CACHE", str(ROOT / ".cache/huggingface/xet"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from huggingface_hub import hf_hub_download

        lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
        print(f"Downloading with Hugging Face: {asset['name']}", flush=True)
        downloaded = Path(hf_hub_download(
            repo_id=lock["model_repository"], filename=asset["name"],
            revision=lock["model_revision"], local_dir=target.parent, token=False,
        ))
        if downloaded != target or digest(downloaded) != asset["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {asset['name']}")
        print(f"Verified: {asset['name']}", flush=True)
        return target
    partial = target.with_name(target.name + ".partial")
    for attempt in range(1, 5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "forte-cash-audit-local-setup"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        url = asset["url"]
        if is_model and MODEL_SOURCE == "modelscope":
            url = modelscope_url(asset)
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                append = offset > 0 and response.status == 206
                count = offset if append else 0
                last_report = count
                with partial.open("ab" if append else "wb") as handle:
                    while chunk := response.read(256 * 1024):
                        handle.write(chunk)
                        count += len(chunk)
                        if count - last_report >= 16 * 1024 * 1024:
                            print(f"{asset['name']}: {count / 1e6:.0f} MB", flush=True)
                            last_report = count
            if digest(partial) != asset["sha256"]:
                raise RuntimeError(f"SHA256 mismatch: {asset['name']}")
            partial.replace(target)
            print(f"Verified: {asset['name']}", flush=True)
            return target
        except RuntimeError:
            raise
        except Exception as error:
            print(f"Retry {attempt}/4: {asset['name']} ({type(error).__name__})", flush=True)
            if attempt == 4:
                raise
            time.sleep(attempt * 2)


def main():
    global MODEL_SOURCE, MODEL_LOCK_PATH, DOWNLOAD_WORKERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-source", choices=["huggingface", "huggingface-range", "modelscope"], default="huggingface-range")
    parser.add_argument("--model-size", choices=["2b", "4b", "8b"], default="2b")
    parser.add_argument("--download-workers", type=int, choices=range(1, 17), default=4,
                        help="Concurrent byte-range requests per model file")
    args = parser.parse_args()
    MODEL_SOURCE = args.model_source
    DOWNLOAD_WORKERS = args.download_workers
    MODEL_LOCK_PATH = ROOT / "analyzer" / (f"runtime-lock-{args.model_size}.json" if args.model_size != "2b" else "runtime-lock.json")
    lock = json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))
    if "runtime_assets_from" in lock:
        runtime_lock = json.loads((MODEL_LOCK_PATH.parent / lock["runtime_assets_from"]).read_text(encoding="utf-8"))
        if runtime_lock["llama_release"] != lock["llama_release"]:
            raise ValueError("Model and runtime locks specify different runtime releases")
        lock["assets"].extend(asset for asset in runtime_lock["assets"] if "extract_to" in asset)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, lock["assets"]))
    # Unpack sequentially because the two archives share a destination.
    for asset in lock["assets"]:
        if "extract_to" not in asset:
            continue
        destination = (ROOT / asset["extract_to"]).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / asset["target"]) as archive:
            for item in archive.infolist():
                if not (destination / item.filename).resolve().is_relative_to(destination):
                    raise ValueError("Archive member escapes runtime directory")
            for item in archive.infolist():
                installed = destination / item.filename
                if not item.is_dir() and installed.is_file() and installed.stat().st_size == item.file_size:
                    with archive.open(item) as source:
                        identical = digest(installed) == hashlib.file_digest(source, "sha256").hexdigest()
                    if identical:
                        # A running Windows server locks its executable. Adding
                        # another model must not rewrite an unchanged runtime.
                        continue
                archive.extract(item, destination)
        print(f"Unpacked: {asset['name']}", flush=True)
    print("Local assets are ready.", flush=True)


if __name__ == "__main__":
    main()
