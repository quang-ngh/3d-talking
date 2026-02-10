import time
import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from smirk.src.smirk_encoder import SmirkEncoder
from smirk.src.FLAME.FLAME import FLAME
from smirk.src.renderer.renderer import Renderer
from smirk.utils.mediapipe_utils import run_mediapipe
from tqdm import tqdm

#   INITIALIZE SMIRK ENCODER
device = "cuda" if torch.cuda.is_available() else "cpu"
smirk_checkpoint_path = 'smirk/pretrained_models/SMIRK_em1.pt'
input_image_size = 224

smirk_encoder = SmirkEncoder().to(device=device)
checkpoint = torch.load(smirk_checkpoint_path)
checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

smirk_encoder.load_state_dict(checkpoint_encoder)
smirk_encoder.eval()

#   INITIALIZE FLAME
# flame = FLAME().to(device=device)
# renderer = Renderer().to(device=device)

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

def process_image(img: np.ndarray, crop: bool = True) -> np.ndarray:
    kpt_mediapipe = run_mediapipe(img)

    if kpt_mediapipe is None:
        print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
        exit()

    if crop:
        kpt_mediapipe = kpt_mediapipe[..., :2]

        tform = crop_face(img,kpt_mediapipe,scale=1.4,image_size=input_image_size)
        
        cropped_img = warp(img, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)
        
        cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
        cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
    else:
        cropped_img = img
        cropped_kpt_mediapipe = kpt_mediapipe
    

    return {
        "cropped_img": cropped_img,
        "cropped_kpt_mediapipe": cropped_kpt_mediapipe,
    }

@torch.inference_mode()
def track_flame_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('Error opening video file')
        exit()
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames_to_process = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames = 0
    num_flame_params = {
        "shape_params": [],
        "expr_params": [],
        "pose_params": [],
        "jaw_params": [],
        "neck_params": [],
        "eye_params": [],
        "translation": [],
    }
    pbar = tqdm(total=num_frames_to_process, unit="frame", desc="Tracking FLAME")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        #   Read frame
        num_frames += 1

        processed_frame = process_image(frame)

        #   Convert to the tensor
        cropped_image = cv2.cvtColor(processed_frame.get("cropped_img", None), cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224,224))
        cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
        cropped_image = cropped_image.to(device=device)

        outputs = smirk_encoder(cropped_image)

        shape_params = outputs.get("shape_params", None)
        expr_params = outputs.get("expression_params", None)
        pose_params = outputs.get("pose_params", None)
        jaw_params = outputs.get("jaw_params", None)
        neck_params = outputs.get("neck_params", None)
        eye_params = outputs.get("eyelid_params", None)

        num_flame_params["shape_params"].append(shape_params.detach().cpu() if shape_params is not None else None)
        num_flame_params["expr_params"].append(expr_params.detach().cpu() if expr_params is not None else None)
        num_flame_params["pose_params"].append(pose_params.detach().cpu() if pose_params is not None else None)
        num_flame_params["jaw_params"].append(jaw_params.detach().cpu() if jaw_params is not None else None)
        num_flame_params["neck_params"].append(neck_params.detach().cpu() if neck_params is not None else None)
        num_flame_params["eye_params"].append(eye_params.detach().cpu() if eye_params is not None else None)

        t_elapsed = time.perf_counter() - t_start
        pbar.set_postfix({"frame_ms": f"{t_elapsed*1000:.1f}", "fps": f"{1/t_elapsed:.1f}" if t_elapsed > 0 else "—"})
        pbar.update(1)

        if num_frames >= 500:
            break
    pbar.close()
    for key, list_params in num_flame_params.items():
        if None in list_params:
            num_flame_params[key] = None
        elif len(list_params) == 0:
            num_flame_params[key] = None
        else:
            num_flame_params[key] = torch.stack(list_params).squeeze(1)

    breakpoint()


    

def main(args):
    video_path = "datasets/processed_hdtf_tfhp/raw_data/data/TH_00005/000.mp4"
    track_flame_video(video_path)

main(args=None)
