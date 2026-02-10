import os 

def get_manifest_hdtf(root_path="./datasets/HDTF/data", save_path="./datasets/manifest/train_hdtf.txt"):
    _sets = ["RD", "WRA", "WDA"]

    results = []
    for split in _sets:
        split_path = split+"_25fps"
        video_dir = os.path.join(root_path, split_path, "videos")
        audio_dir = os.path.join(root_path, split_path, "wav")
        for video in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video)
            audio_path = os.path.join(audio_dir, video.replace(".mp4", ".wav"))
            results.append(f"{video_path} {audio_path}")
    with open(save_path, "w") as f:
        for result in results:
            f.write(result + "\n")

if __name__ == "__main__":
    get_manifest_hdtf()