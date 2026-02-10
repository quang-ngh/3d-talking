#!/usr/bin/env python3
"""
Download YouTube videos from annotations in sft_split.json using yt-dlp.

Extracts unique video IDs from the annotation file (key: yt_id by default)
and downloads each video once to an output directory.
Supports downloading only a specific time range (segment) per video.

Usage:
  python download_youtube_from_sft.py
  python download_youtube_from_sft.py --annotation sft_split.json --out-dir videos
  python download_youtube_from_sft.py --start 1:30 --end 2:45
  python download_youtube_from_sft.py --start 90 --duration 75

Requires: pip install yt-dlp (or pip install "yt-dlp[default]"), ffmpeg (for --download-sections).
YouTube also requires a JS runtime (Deno 2+ or Node 20+) and EJS scripts; the script uses
--remote-components ejs:github by default. See https://github.com/yt-dlp/yt-dlp/wiki/EJS
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Lock for serializing progress prints from workers
_print_lock = threading.Lock()


def parse_time_to_seconds(s: str) -> float:
    """Parse time string (seconds, M:SS, or HH:MM:SS) to seconds."""
    s = s.strip()
    if re.match(r"^\d+\.?\d*$", s):
        return float(s)
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    raise ValueError(f"Invalid time format: {s!r}. Use seconds, M:SS, or HH:MM:SS")


def seconds_to_timestamp(sec: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS for yt-dlp."""
    sec = max(0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if s == int(s):
        s_int = int(s)
        if h > 0:
            return f"{h}:{m:02d}:{s_int:02d}"
        return f"{m}:{s_int:02d}"
    if h > 0:
        return f"{h}:{m:02d}:{s:06.3f}"
    return f"{m}:{s:06.3f}"


def load_video_ids(
    annotation_path: str = "",
    id_key: str = "yt_id",
) -> list[str]:
    """Load and return unique video IDs from annotation JSON."""
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Annotation file must contain a list of dicts")

    seen = set()
    ids = []
    for item in data:
        if not isinstance(item, dict):
            continue
        vid = item.get(id_key)
        if vid and vid not in seen:
            seen.add(vid)
            ids.append(vid)
    return ids


def download_video(
    video_id: str,
    out_dir: Path,
    format: str = "(bestvideo[ext=mp4]/bestvideo)+bestaudio[ext=m4a]/best[ext=mp4]/best",
    cookies: str = "",
    section_start: float | None = None,
    section_end: float | None = None,
    js_runtime: str | None = None,
    remote_components: str | None = "ejs:github",
) -> bool:
    """Download a single YouTube video by ID, optionally only a time range. Returns True on success."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if section_start is not None and section_end is not None:
        # Segment download: use section template so filename is unique
        out_tmpl = str(out_dir / "%(id)s_%(section_start)s_%(section_end)s.%(ext)s")
        section_spec = f"*{seconds_to_timestamp(section_start)}-{seconds_to_timestamp(section_end)}"
    else:
        out_tmpl = str(out_dir / "%(id)s.%(ext)s")
        section_spec = None

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-overwrites",
        "-o",
        out_tmpl,
        "-f",
        format,
        url,
    ]
    if section_spec:
        cmd.extend(["--download-sections", section_spec])
    if cookies:
        cmd.extend(["--cookies", cookies])
    # Required for YouTube "n challenge" solving (see https://github.com/yt-dlp/yt-dlp/wiki/EJS)
    if remote_components:
        cmd.extend(["--remote-components", remote_components])
    if js_runtime:
        cmd.extend(["--js-runtimes", js_runtime])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed {video_id}: {e.stderr or e}", file=sys.stderr)
        return False


def _download_chunk(
    worker_id: int,
    chunk: list[str],
    out_dir: Path,
    format: str,
    cookies: str | None,
    section_start: float | None,
    section_end: float | None,
    js_runtime: str | None,
    remote_components: str | None,
) -> int:
    """Download a chunk of videos; returns count of successful downloads."""
    total = len(chunk)
    ok = 0
    for idx, video_id in enumerate(chunk, 1):
        with _print_lock:
            pct = 100 * idx / total if total else 0
            print(f"[Worker {worker_id}] {idx}/{total} ({pct:.1f}%) - {video_id}", flush=True)
        if download_video(
            video_id,
            out_dir,
            format=format,
            cookies=cookies or "",
            section_start=section_start,
            section_end=section_end,
            js_runtime=js_runtime,
            remote_components=remote_components,
        ):
            ok += 1
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube videos from sft_split.json using yt-dlp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--annotation",
        default="sft_split.json",
        help="Path to annotation JSON (default: sft_split.json)",
    )
    parser.add_argument(
        "--id-key",
        default="yt_id",
        help="Key in each dict that holds the YouTube video ID (default: yt_id)",
    )
    parser.add_argument(
        "--out-dir",
        default="youtube_videos",
        help="Directory to save downloaded videos (default: youtube_videos)",
    )
    parser.add_argument(
        "--format",
        default="(bestvideo[ext=mp4]/bestvideo)+bestaudio[ext=m4a]/best[ext=mp4]/best",
        help="yt-dlp format string (default: prefer MP4, fallback to best)",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to Netscape cookies file for age-restricted or region-locked videos",
    )
    parser.add_argument(
        "--js-runtime",
        default=None,
        choices=["node", "deno", "bun", "quickjs"],
        help="JavaScript runtime for YouTube challenge solving (default: deno if available). Use 'node' if you have Node.js 20+.",
    )
    parser.add_argument(
        "--no-ejs-remote",
        action="store_true",
        help="Disable EJS script download (e.g. if using pip install 'yt-dlp[default]')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of videos to download (default: all)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Max videos per worker chunk (default: 5000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Max parallel workers (default: min(cpu_count, num_chunks))",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print video IDs that would be downloaded",
    )
    # Time range (segment) - requires ffmpeg
    parser.add_argument(
        "--start",
        default=None,
        metavar="TIME",
        help="Start time of segment (seconds, or M:SS, or HH:MM:SS). Use with --end or --duration.",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="TIME",
        help="End time of segment (seconds, or M:SS, or HH:MM:SS). Ignored if --duration is set.",
    )
    parser.add_argument(
        "--duration",
        default=None,
        metavar="SECONDS",
        help="Duration of segment in seconds (end = start + duration). Overrides --end.",
    )
    args = parser.parse_args()

    section_start = None
    section_end = None
    if args.start is not None:
        section_start = parse_time_to_seconds(args.start)
        if args.duration is not None:
            section_end = section_start + float(args.duration)
        elif args.end is not None:
            section_end = parse_time_to_seconds(args.end)
        else:
            print("Error: --start requires --end or --duration", file=sys.stderr)
            sys.exit(1)
        if section_end <= section_start:
            print("Error: end time must be after start time", file=sys.stderr)
            sys.exit(1)
        print(f"Downloading segment: {seconds_to_timestamp(section_start)} to {seconds_to_timestamp(section_end)}")

    try:
        video_ids = load_video_ids(args.annotation, id_key=args.id_key)
    except Exception as e:
        print(f"Error loading annotations: {e}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        video_ids = video_ids[: args.limit]

    print(f"Found {len(video_ids)} unique video(s) from {args.annotation}")

    if args.dry_run:
        for vid in video_ids:
            print(f"  https://www.youtube.com/watch?v={vid}")
        return

    out_dir = Path(args.out_dir)
    remote_components = None if args.no_ejs_remote else "ejs:github"

    print(f"Format: {args.format}")

    # Partition into chunks of --chunk-size (default 5000) per worker
    chunk_size = max(1, args.chunk_size)
    chunks = [
        video_ids[i : i + chunk_size]
        for i in range(0, len(video_ids), chunk_size)
    ]
    num_chunks = len(chunks)
    cpu_count = os.cpu_count() or 4
    max_workers = args.workers
    if max_workers is None:
        max_workers = min(cpu_count, num_chunks)
    max_workers = max(1, min(max_workers, num_chunks))

    print(f"Total videos: {len(video_ids)} | Chunks: {num_chunks} (max {chunk_size}/chunk) | Workers: {max_workers}")
    print("-" * 60)

    ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_chunk,
                worker_id,
                chunk,
                out_dir,
                args.format,
                args.cookies,
                section_start,
                section_end,
                args.js_runtime,
                remote_components,
            ): worker_id
            for worker_id, chunk in enumerate(chunks, 1)
        }
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                n = future.result()
                with _print_lock:
                    print(f"[Worker {worker_id}] finished: {n} succeeded", flush=True)
                ok += n
            except Exception as e:
                with _print_lock:
                    print(f"[Worker {worker_id}] error: {e}", file=sys.stderr, flush=True)

    print("-" * 60)
    print(f"Done. {ok}/{len(video_ids)} downloaded to {out_dir}")


if __name__ == "__main__":
    main()
