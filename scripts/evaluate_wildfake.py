"""Evaluate a trained checkpoint against a COCO2017-real vs WildFake-fake
sample, mirroring the composition another team (huythuan-bui/model_training)
used for their reported results: 1,200 real COCO2017 images vs 1,200 fake
images spread across several diffusion generators, 200 each.

EVAL-ONLY. This script is deliberately kept out of detector/data_sources/ and
out of config.yaml's data_source enum -- it must never be reachable from
`pipeline.py train`. It only ever reads images into a throwaway temp
directory to run detector.evaluation.evaluate() against an already-trained
checkpoint.

WildFake (arXiv 2402.11843, AAAI 2025) has no confirmed public HF/GitHub
release; it is hosted on ModelScope (https://modelscope.cn/datasets/
hy2628982280/WildFake), not HuggingFace -- which is why detector/
data_sources/dragon.py's HfFileSystem approach doesn't apply here. The full
dataset is ~1.2TB (individual generator zips range 6-53GB), so this streams
only the ~2,200 needed images directly out of the remote zips via HTTP range
requests instead of downloading whole archives -- see
_BufferedHTTPRangeFile below.

Generator coverage: the reference screenshot showed 6 generators (ddim,
ddpm, dalle, vqdm, adm, +1 cut off by the screenshot's keyboard overlay).
This script covers the 5 confirmed, single-zip generators (ADM, DALLE,
DDIM, DDPM, VQDM) -- each has one clean Images/Diffusion_based/<NAME>.zip.
Midjourney and SD were left out: both are split across many 50GB+ part_*.zip
files per (Advanced/Typical) bucket, and the test-split CSVs don't indicate
which part a given image lives in, so covering them needs a beefier
byte-range-probing pass this deadline didn't leave room for.

Usage:
    ./.venv/Scripts/python.exe scripts/evaluate_wildfake.py \\
        --checkpoint checkpoints/detector.pt --per-generator 200 --real-count 1200
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import zipfile
from pathlib import Path
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.data import ImageRecord  # noqa: E402
from detector.data_sources._network import retry_network_call  # noqa: E402
from detector.evaluation import evaluate  # noqa: E402

MODELSCOPE_API = "https://modelscope.cn/api/v1/datasets/hy2628982280/WildFake/repo"

# ModelScope's CDN is measurably hostile to a long-lived pooled connection
# here. The first smoke test hung on its very first byte-range probe and
# burned ~24 minutes retrying, while `curl` -- a fresh process, so a fresh
# connection, every time -- served the identical request in 3.7s. The client
# had just pulled a 12MB CSV, so the pooled keep-alive socket was almost
# certainly closed server-side without httpx knowing, and every read then
# blocked until the full timeout.
#
# max_keepalive_connections=0 forces a new connection per request. That costs
# a TLS handshake each time and is the wrong default for a chatty API, but
# this script makes tens of large reads rather than thousands of small ones,
# so the handshake is noise against the transfer.
#
# The User-Agent is not about being allowed in -- every UA tested returned
# 206 -- it is about speed: python-httpx's default measured 19.4s against
# 3.4s for a browser UA on the same request, repeatably.
_CLIENT_LIMITS = httpx.Limits(max_keepalive_connections=0, max_connections=4)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Encoding": "identity",
}

# name -> (zip path, test-split CSV path with an Image_path column, image path
# prefix to strip -- CSV rows look like "./Diffusion_based/ADM/imgs/xyz.png"
# and the zip's own internal member paths match that verbatim).
FAKE_GENERATORS: dict[str, tuple[str, str]] = {
    "adm": ("Images/Diffusion_based/ADM.zip", "split_train_test/csv_file/cross_architectures/ADM_test.csv"),
    "dalle": ("Images/Diffusion_based/DALLE.zip", "split_train_test/csv_file/cross_architectures/DALLE_test.csv"),
    "ddim": ("Images/Diffusion_based/DDIM.zip", "split_train_test/csv_file/cross_architectures/DDIM_test.csv"),
    "ddpm": ("Images/Diffusion_based/DDPM.zip", "split_train_test/csv_file/cross_architectures/DDPM_test.csv"),
    "vqdm": ("Images/Diffusion_based/VQDM.zip", "split_train_test/csv_file/cross_architectures/VQDM_test.csv"),
}
REAL_ZIP = "Images/Real/coco.zip"
REAL_CSV = "label_csv_files/real_coco.csv"


def _file_url(path: str) -> str:
    return f"{MODELSCOPE_API}?Revision=master&FilePath={quote(path, safe='')}"


class _BufferedHTTPRangeFile:
    """Minimal seekable file-like object over one HTTP resource, read via
    Range requests, with a large read-ahead buffer so zipfile's usual
    pattern of many small reads (central directory parsing, then one
    seek+read per extracted member) turns into a handful of HTTP round
    trips per zip instead of thousands.
    """

    # 1MB read-ahead: the images here run ~100-400KB, so a larger chunk
    # mostly buys bytes that get thrown away on the next seek. Reads larger
    # than the chunk are handled (read() fetches max(n, chunk_size)), so this
    # is a throughput knob, not a correctness one.
    def __init__(self, client: httpx.Client, url: str, *, chunk_size: int = 1024 * 1024) -> None:
        self._client = client
        self._url = url
        self._chunk_size = chunk_size
        self._pos = 0
        self._buf = b""
        self._buf_start = 0

        def _probe() -> httpx.Response:
            resp = client.get(url, headers={"Range": "bytes=0-0"}, follow_redirects=True)
            resp.raise_for_status()
            return resp

        # Fewer/faster retries than the training-time default (8 x 60s): a
        # multi-hour training run is worth ~7 minutes of patience, an eval
        # script is not -- and the failure mode here (stale connection) is
        # fixed by reconnecting immediately, not by waiting a minute.
        resp = retry_network_call(
            _probe, description=f"probe size of {url}", attempts=4, delay_seconds=5.0
        )
        content_range = resp.headers.get("Content-Range", "")
        self._size = int(content_range.rsplit("/", 1)[-1])

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, n: int | None = -1) -> bytes:
        if n is None or n < 0:
            n = self._size - self._pos
        n = max(0, min(n, self._size - self._pos))
        if n == 0:
            return b""

        buf_end = self._buf_start + len(self._buf)
        if self._pos >= self._buf_start and self._pos + n <= buf_end:
            start = self._pos - self._buf_start
            data = self._buf[start : start + n]
            self._pos += len(data)
            return data

        fetch_len = max(n, self._chunk_size)
        end = min(self._pos + fetch_len, self._size) - 1

        def _fetch() -> httpx.Response:
            resp = self._client.get(
                self._url,
                headers={"Range": f"bytes={self._pos}-{end}"},
                follow_redirects=True,
            )
            resp.raise_for_status()
            return resp

        resp = retry_network_call(
            _fetch,
            description=f"range GET {self._url} [{self._pos}-{end}]",
            attempts=4,
            delay_seconds=5.0,
        )
        self._buf = resp.content
        self._buf_start = self._pos
        data = self._buf[:n]
        self._pos += len(data)
        return data


def _fetch_csv_rows(client: httpx.Client, path: str) -> list[dict[str, str]]:
    def _get() -> httpx.Response:
        resp = client.get(_file_url(path), follow_redirects=True)
        resp.raise_for_status()
        return resp

    resp = retry_network_call(_get, description=f"GET {path}", attempts=4, delay_seconds=5.0)
    reader = csv.DictReader(io.StringIO(resp.text))
    return list(reader)


def _sample_image_paths(
    rows: list[dict[str, str]], count: int, seed: int, *, is_fake: str
) -> list[str]:
    """Sample `count` image paths of one class from a WildFake split CSV.

    The ``is_fake`` filter is load-bearing, not defensive. WildFake's
    per-architecture test CSVs are *pre-mixed*: ADM_test.csv carries 31,005
    fake rows and 15,503 real ones (their repo has add_real_cross_*.py
    scripts that append a real set to each). Sampling without filtering
    silently drew ~1/3 real images into the fake half -- those were skipped
    rather than mislabelled, because they live in a different zip, so the
    only visible symptom was a stream of "not found in ADM.zip" warnings and
    a fake sample quietly ~35% smaller than requested.
    """
    rng = random.Random(seed)
    pool = [
        row["Image_path"].lstrip("./")
        for row in rows
        if row.get("IsFake", "").strip() == is_fake
    ]
    if len(pool) < count:
        raise ValueError(
            f"Only {len(pool)} rows with IsFake={is_fake} available, need {count}."
        )
    return rng.sample(pool, count)


def _extract_images(
    client: httpx.Client,
    zip_path: str,
    member_paths: list[str],
    dest_dir: Path,
    label: int,
) -> list[ImageRecord]:
    dest_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(member: str) -> Path:
        return dest_dir / (hashlib.sha256(member.encode()).hexdigest()[:16] + Path(member).suffix)

    # Reuse anything already extracted. Pulling 1,000 images out of a 2.3GB
    # remote zip takes ~55 minutes of range requests, and re-running the
    # script after an unrelated fix should not pay that again. The filename
    # is a hash of the member path, so a file that is present is necessarily
    # the right image. Checked before opening the archive, so a fully cached
    # set touches the network zero times.
    records: list[ImageRecord] = []
    outstanding: list[str] = []
    for member in member_paths:
        cached = _cache_path(member)
        if cached.exists() and cached.stat().st_size > 0:
            records.append(
                ImageRecord(
                    path=cached,
                    label=label,
                    sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
                    split="wildfake_eval",
                )
            )
        else:
            outstanding.append(member)

    if records:
        print(f"    reused {len(records)} cached, fetching {len(outstanding)}")
    if not outstanding:
        return records

    remote = _BufferedHTTPRangeFile(client, _file_url(zip_path))
    with zipfile.ZipFile(remote) as archive:
        names = set(archive.namelist())

        # Read in archive order, not sample order. Which images are in the
        # sample was already decided (randomly) by _sample_image_paths; this
        # only changes the order they are fetched in, so it cannot bias the
        # set. It makes every seek forward-only, so the read-ahead buffer is
        # occasionally reused instead of being discarded on every jump
        # backwards -- which matters when one full run is ~2,000 range reads
        # over multi-GB archives.
        offsets = {}
        for member in outstanding:
            try:
                offsets[member] = archive.getinfo(member).header_offset
            except KeyError:
                offsets[member] = 0
        outstanding = sorted(outstanding, key=lambda m: offsets[m])

        for member in outstanding:
            if member not in names:
                # WildFake's zip members are occasionally stored with a
                # leading "./" the CSV strips or a different case; try both
                # before giving up on this one image.
                candidates = [n for n in names if n.endswith(member.split("/")[-1])]
                if not candidates:
                    print(f"  [WARN] {member} not found in {zip_path}, skipping")
                    continue
                member = candidates[0]
            data = archive.read(member)
            out_path = dest_dir / (hashlib.sha256(member.encode()).hexdigest()[:16] + Path(member).suffix)
            out_path.write_bytes(data)
            records.append(
                ImageRecord(
                    path=out_path,
                    label=label,
                    sha256=hashlib.sha256(data).hexdigest(),
                    split="wildfake_eval",
                )
            )
    return records


def _final_score(clean_auc: float | None, robust_auc: float | None) -> float | None:
    if clean_auc is None or robust_auc is None:
        return None
    return 0.5 * clean_auc + 0.5 * robust_auc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="path to a trained checkpoint (.pt)")
    parser.add_argument("--per-generator", type=int, default=200)
    parser.add_argument("--real-count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache-dir", default=None, help="defaults to a temp dir under runs/")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir) if args.cache_dir else Path("runs") / "wildfake_eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        timeout=60.0, limits=_CLIENT_LIMITS, headers=_HEADERS, follow_redirects=True
    ) as client:
        print("[WILDFAKE] fetching real_coco.csv ...")
        real_rows = _fetch_csv_rows(client, REAL_CSV)
        real_paths = _sample_image_paths(
            real_rows, args.real_count, seed=args.seed, is_fake="0"
        )
        print(f"[WILDFAKE] extracting {len(real_paths)} real COCO images (this is the slow step; single zip)")
        real_records = _extract_images(client, REAL_ZIP, real_paths, cache_dir / "real", label=0)

        per_generator_records: dict[str, list[ImageRecord]] = {}
        for name, (zip_path, csv_path) in FAKE_GENERATORS.items():
            print(f"[WILDFAKE] fetching {name} test-split CSV ...")
            rows = _fetch_csv_rows(client, csv_path)
            paths = _sample_image_paths(
                rows, args.per_generator, seed=args.seed, is_fake="1"
            )
            print(f"[WILDFAKE] extracting {len(paths)} {name} images ...")
            per_generator_records[name] = _extract_images(client, zip_path, paths, cache_dir / name, label=1)

    real_subset_for_rows = real_records[: args.per_generator]

    results: dict[str, dict] = {}
    print("\n[EVALUATE] per-generator ...")
    for name, fake_records in per_generator_records.items():
        summary = evaluate(
            checkpoint=args.checkpoint,
            records=real_subset_for_rows + fake_records,
            output_dir=str(cache_dir / f"eval_{name}"),
            device=args.device,
        )
        auc_clean = summary["clean"]["roc_auc"]
        auc_robust = summary["robust_mean"]["roc_auc"]
        results[name] = {
            "auc_clean": auc_clean,
            "auc_robust": auc_robust,
            "final_score": _final_score(auc_clean, auc_robust),
        }
        print(f"  {name:8s} AUC_clean={auc_clean:.4f}  AUC_robust={auc_robust:.4f}")

    print("\n[EVALUATE] aggregate (all generators + full real pool) ...")
    all_fake = [r for records in per_generator_records.values() for r in records]
    aggregate_summary = evaluate(
        checkpoint=args.checkpoint,
        records=real_records + all_fake,
        output_dir=str(cache_dir / "eval_aggregate"),
        device=args.device,
    )
    agg_clean = aggregate_summary["clean"]["roc_auc"]
    agg_robust = aggregate_summary["robust_mean"]["roc_auc"]
    agg_final = _final_score(agg_clean, agg_robust)

    report = {
        "composition": f"COCO2017 authentic ({len(real_records)}) vs WildFake AI "
        f"({len(all_fake)}) -- {len(per_generator_records)} generators, {args.per_generator} each",
        "final_score": agg_final,
        "auc_clean": agg_clean,
        "auc_robust": agg_robust,
        "per_generator": results,
    }
    report_path = cache_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Results on the WildFake composition ===")
    print(f"Final Score  {agg_final:.4f}")
    print(f"AUC_clean    {agg_clean:.4f}")
    print(f"AUC_robust   {agg_robust:.4f}")
    print(f"\n{'Generator':10s} {'AUC clean':>10s} {'AUC robust':>11s}")
    for name, row in results.items():
        print(f"{name:10s} {row['auc_clean']:>10.3f} {row['auc_robust']:>11.3f}")
    print(f"\nFull report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
