"""Pin public model revisions and verify every downloaded file before local use."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

REPOS = {
    "chronos": "amazon/chronos-t5-base",
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "sundial": "thuml/sundial-base-128m",
    "moirai": "Salesforce/moirai-1.1-R-base",
}


def verify(path, item):
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    h = hashlib.sha256() if "lfs" in item else hashlib.sha1()
    if "lfs" not in item:
        h.update(f"blob {item['size']}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest() == (item["lfs"]["sha256"] if "lfs" in item else item["blobId"])


def download(url, path, item):
    if verify(path, item):
        return
    if path.exists():
        raise ValueError(f"Existing artifact failed verification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(6):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset < item["size"]:
                req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
                with urllib.request.urlopen(req, timeout=120) as response:
                    if offset and (
                        response.status != 206
                        or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                    ):
                        raise RuntimeError("Resume range not honored")
                    with part.open("ab" if offset else "wb") as out:
                        while block := response.read(4 * 1024**2):
                            out.write(block)
            if not verify(part, item):
                raise ValueError(f"Size/hash verification failed: {part}")
            part.replace(path)
            print(f"Verified {path} ({item['size']} bytes)", flush=True)
            return
        except (OSError, RuntimeError) as exc:
            if attempt == 5:
                raise
            print(f"Retry {path.name}: {exc}", flush=True)
            time.sleep(5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, repo in REPOS.items():
        metadata_path = root / f"{name}.source.json"
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            with urllib.request.urlopen(
                f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=90
            ) as response:
                meta = json.load(response)
            metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        checkpoint = root / "weights" / name
        files = []
        for item in meta["siblings"]:
            file = item["rfilename"]
            if not file.endswith((".json", ".safetensors", ".py", "README.md", "LICENSE", "LICENSE.txt")):
                continue
            path = (checkpoint / file).resolve()
            if not path.is_relative_to(checkpoint.resolve()):
                raise ValueError("Checkpoint path traversal")
            download(f"https://huggingface.co/{repo}/resolve/{meta['sha']}/{file}", path, item)
            files.append(item)
        if not any(x["rfilename"].endswith(".safetensors") for x in files):
            raise ValueError(f"No safetensors weights: {repo}")
        records[name] = dict(repo=repo, revision=meta["sha"], checkpoint=str(checkpoint), files=files)
        (root / "weights_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Pinned {name}: {meta['sha']}", flush=True)


if __name__ == "__main__":
    main()
