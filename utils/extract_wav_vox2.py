import subprocess
import os
from omegaconf import OmegaConf
from tqdm import tqdm

def extract_wav(input_dir, output_dir, start_idx=0, end_idx=-1):
    list_files = sorted(os.listdir(input_dir))
    if end_idx < 0:
        end_idx = len(list_files)

    for file in tqdm(list_files[start_idx:end_idx]):
        if file.endswith(".mp4"):
            subprocess.run([
                "ffmpeg", "-i", 
                os.path.join(input_dir, file), 
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                os.path.join(output_dir, file.replace(".mp4", ".wav"))])

def extract_wav_vox2(root_dir, start_idx=0, end_idx=-1):
    list_speakers = os.listdir(root_dir)
    for speaker in tqdm(list_speakers[start_idx:end_idx]):
        dir_speaker = os.path.join(root_dir, speaker)
        extract_wav(dir_speaker, dir_speaker)
    
if __name__ == "__main__":
    args = OmegaConf.from_cli()
    extract_wav_vox2(args.root_dir, args.start_idx, args.end_idx)