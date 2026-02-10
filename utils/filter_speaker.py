#!/usr/bin/env python3
"""
Filter manifest for sft_split with single person talking.
Uses multi-threading for faster processing.

Usage:
    python filter_speaker.py --workers 8
    python filter_speaker.py --output filtered_sft_split.json
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Default paths
MERGED_ANNO_PATH = "/common/users/hn315/datasets/speaker5M/merged_anno"
SFT_SPLIT_PATH = "../talking-head/datasets/speaker5M/sft_split.json"


def load_sft_split(sft_split_path: str) -> List[Dict[str, Any]]:
    """Load the sft_split JSON file."""
    with open(sft_split_path, "r") as f:
        return json.load(f)


def check_single_speaker(json_path: str) -> Tuple[str, bool, Optional[Dict]]:
    """
    Check if a JSON file contains single person talking.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Tuple of (filename, is_valid, data)
    """
    filename = os.path.basename(json_path)
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Check if single person talking
        is_valid = (
            data.get("is_talking", False) and 
            data.get("clip_speaker_num", 999) < 2
        )
        
        return (filename, is_valid, data if is_valid else None)
    
    except Exception as e:
        return (filename, False, None)


def filter_single_speaker_multithread(
    merged_anno_path: str,
    sft_split: List[Dict[str, Any]],
    workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Filter for single speaker files using multiple threads.
    
    Args:
        merged_anno_path: Path to directory containing annotation JSONs
        sft_split: List of items from sft_split.json
        workers: Number of parallel workers
        
    Returns:
        List of filtered items (single speaker only)
    """
    # Build set of valid filenames from sft_split
    valid_filenames = {item["filename"] for item in sft_split}
    
    # Get all JSON files that are in sft_split
    all_json_files = os.listdir(merged_anno_path)
    json_files_to_check = [
        os.path.join(merged_anno_path, f) 
        for f in all_json_files 
        if f in valid_filenames
    ]
    
    print(f"Total files in sft_split: {len(valid_filenames)}")
    print(f"Files to check: {len(json_files_to_check)}")
    
    # Process files in parallel
    keep_filenames = set()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(check_single_speaker, path): path 
            for path in json_files_to_check
        }
        
        with tqdm(total=len(futures), desc="Filtering", unit="file") as pbar:
            for future in as_completed(futures):
                filename, is_valid, _ = future.result()
                
                if is_valid:
                    keep_filenames.add(filename)
                
                pbar.update(1)
                pbar.set_postfix({"kept": len(keep_filenames)}, refresh=False)
    
    # Filter original sft_split to keep only single speaker items
    filtered_items = [
        item for item in sft_split 
        if item["filename"] in keep_filenames
    ]
    
    print(f"\nFiltered: {len(filtered_items)} / {len(sft_split)} items kept")
    
    return filtered_items


def single_thread_filter(
    merged_anno_path: str,
    sft_split: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Original single-threaded filter (for reference/comparison).
    """
    valid_filenames = {item["filename"] for item in sft_split}
    keep_filenames = set()
    
    all_json_files = os.listdir(merged_anno_path)
    
    for json_file in tqdm(all_json_files, desc="Filtering"):
        if json_file not in valid_filenames:
            continue
            
        json_path = os.path.join(merged_anno_path, json_file)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            if data.get("is_talking", False) and data.get("clip_speaker_num", 999) < 2:
                keep_filenames.add(json_file)
        except Exception:
            pass
    
    filtered_items = [
        item for item in sft_split 
        if item["filename"] in keep_filenames
    ]
    
    return filtered_items


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter sft_split for single person talking"
    )
    
    parser.add_argument(
        "--merged_anno_path",
        type=str,
        default=MERGED_ANNO_PATH,
        help=f"Path to merged annotations (default: {MERGED_ANNO_PATH})",
    )
    parser.add_argument(
        "--sft_split_path",
        type=str,
        default=SFT_SPLIT_PATH,
        help=f"Path to sft_split.json (default: {SFT_SPLIT_PATH})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="filtered_sft_split.json",
        help="Output path for filtered manifest (default: filtered_sft_split.json)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--single_thread",
        action="store_true",
        help="Use single-threaded processing (slower)",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="filtered_filenames.txt",
        help="Output path for filenames txt (default: filtered_filenames.txt)",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading sft_split from: {args.sft_split_path}")
    sft_split = load_sft_split(args.sft_split_path)
    print(f"Loaded {len(sft_split)} items")
    
    print(f"Merged anno path: {args.merged_anno_path}")
    print(f"Workers: {args.workers}")
    print()
    
    if args.single_thread:
        filtered_items = single_thread_filter(
            args.merged_anno_path,
            sft_split,
        )
    else:
        filtered_items = filter_single_speaker_multithread(
            args.merged_anno_path,
            sft_split,
            workers=args.workers,
        )
    
    # Save filtered manifest (JSON)
    print(f"\nSaving filtered manifest to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(filtered_items, f, indent=2)
    
    # Save filenames to txt (one per line)
    print(f"Saving filenames to: {args.output_txt}")
    with open(args.output_txt, "w") as f:
        for item in filtered_items:
            f.write(item["filename"] + "\n")
    
    print(f"Done! Kept {len(filtered_items)} single-speaker items.")


if __name__ == "__main__":
    main()
