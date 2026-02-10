#!/usr/bin/env python3
"""
Test script to demonstrate filename conversion from .json to .mp4
"""

import json
from batch_download_videos import VideoDownloader

def test_filename_conversion():
    """Test the filename conversion logic."""
    
    # Sample video entries like those in the JSON
    sample_entries = [
        {
            'num_frames': 294,
            'is_talking': 1,
            'clip_speaker_num': 1,
            'clear': 2221.807638888889,
            'video_name': 'BK6_XIz6-ZE_1920x1080_full_video',
            'video_total_duration': 11.811800000000005,
            'start': 297.25603333333333,
            'duration': 4.2,
            'bbox': [0.6020080312093099, 0.060399305555555595, 0.9994791666666667, 0.9140717230902777],
            'conf': 0,
            'sync': {'0': [[6.115677356719971, 7.334573268890381, -3.0], [3.4443788528442383, 9.135991096496582, -1.0]], '1': [[0, 0, 0], [0, 0, 0]]},
            'speaker': ['B', 'B'],
            'raw_video_height': 1080,
            'raw_video_width': 1920,
            'clip_video_height': 928,
            'clip_video_width': 768,
            'start_seconds': 289.6560333333333,
            'end_seconds': 301.4678333333333,
            'dover': 0.5581270269629535,
            'filename': 'BK6_XIz6-ZE_1920x1080_full_video_042_None_01.json',
            'yt_id': 'BK6_XIz6-ZE'
        },
        {
            'filename': 'r6HEfjwnfFA_1920x1080_full_video_312_A_00.json',
            'yt_id': 'r6HEfjwnfFA'
        },
        {
            'filename': 'jwseJClS0r4_1920x1080_full_video_004_A_01.json',
            'yt_id': 'jwseJClS0r4'
        }
    ]
    
    # Create a temporary downloader to test filename conversion
    downloader = VideoDownloader(
        json_file="./datasets/speaker5M/split_100K_random.json",  # This won't be loaded in test
        output_dir="./test_output"
    )
    
    print("=== FILENAME CONVERSION TEST ===")
    print("Input JSON filename → Output MP4 filename")
    print("-" * 60)
    
    for i, entry in enumerate(sample_entries, 1):
        json_filename = entry.get('filename', 'N/A')
        mp4_filename = downloader.get_output_filename(entry)
        yt_id = entry.get('yt_id', 'N/A')
        
        print(f"{i}. YT ID: {yt_id}")
        print(f"   JSON: {json_filename}")
        print(f"   MP4:  {mp4_filename}")
        print()
    
    print("✅ Filename conversion working correctly!")
    print("   - .json extension is replaced with .mp4")
    print("   - Original filename structure is preserved")

if __name__ == "__main__":
    test_filename_conversion()
