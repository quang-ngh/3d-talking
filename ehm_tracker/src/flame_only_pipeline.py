"""
FLAME-only tracking pipeline - extracts and optimizes FLAME parameters from video.
Simplified version without body/hand tracking.
"""
import os
import cv2
import torch
import imageio
import numpy as np
import os.path as osp
import math
import json
from tqdm.auto import tqdm
from PIL import Image

from .utils.rprint import rlog as log
from .utils.video import images2video
from .utils.lmdb import LMDBEngine
from .utils.io import load_config, write_dict_pkl, load_dict_pkl
from .utils.crop import crop_image
from .utils.helper import load_onnx_model, instantiate_from_config, image2tensor
from .utils.landmark_runner import LandmarkRunner
from .modules.dwpose import inference_detector
from .modules.refiner.flame_refiner import FlameOptimizer
from .modules.renderer.util import cam2persp_cam_fov
from pytorch3d.renderer import PointLights


class FlameOnlyPipeline:
    """Pipeline for FLAME-only head tracking and optimization."""
    
    def __init__(self, config):
        self.cfg = config
        self.device = f'cuda:{self.cfg.device_id}'
        
        log("Loading models...")
        
        # DWPose for face detection
        dwpose_cfg = load_config('src/configs/model_configs/dwpose_onnx_config.yaml')
        self.dwpose_detector = instantiate_from_config(dwpose_cfg)
        self.dwpose_detector.warmup()
        
        # Face landmarks
        self.landmark_runner = LandmarkRunner(
            ckpt_path='pretrained/lmk106/landmark.onnx',
            onnx_provider=self.device
        )
        self.landmark_runner.warmup()
        
        lmk70_cfg = load_config('src/configs/model_configs/lmk70_onnx_config.yaml')
        self.lmk70_detector = load_onnx_model(lmk70_cfg)
        
        mp_cfg = load_config('src/configs/model_configs/mediapipe_detector.yaml')
        self.mp_detector = instantiate_from_config(mp_cfg)
        self.mp_detector.warmup()
        
        # TEASER encoder for FLAME
        teaser_cfg = load_config('src/configs/model_configs/teaser_onnx_config.yaml')
        self.teaser_encoder = load_onnx_model(teaser_cfg)
        
        # FLAME optimizer
        if not self.cfg.skip_optimization:
            self.flame_opt = FlameOptimizer(
                'assets/FLAME',
                device=self.device,
                image_size=512,
                tanfov=1/24
            )
            self.flame = self.flame_opt.flame
            self.renderer = self.flame_opt.renderer
        
        log("✓ Models loaded")
    
    def get_video_name(self, video_fp):
        """Extract clean video name from path."""
        name = osp.basename(video_fp)
        if name.endswith('.mp4'):
            name = name[:-4]
        return name
    
    def get_union_face_box(self, video_fp, frame_interval=1):
        """Get union bounding box for face across sampled frames."""
        reader = imageio.get_reader(video_fp)
        num_frames = reader.count_frames()
        face_boxes = []
        
        for idx in tqdm(range(0, num_frames, frame_interval), desc='Computing face bbox'):
            img_rgb = reader.get_data(idx)
            det_info, _ = self.dwpose_detector(img_rgb)
            if det_info['faces'] is not None:
                # Compute bbox from face keypoints
                face_kps = det_info['faces']
                x_coords = face_kps[:, 0]
                y_coords = face_kps[:, 1]
                x1, y1 = x_coords.min(), y_coords.min()
                x2, y2 = x_coords.max(), y_coords.max()
                
                # Expand bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                size = max(x2 - x1, y2 - y1) * 1.75  # scale factor
                x1, y1 = cx - size / 2, cy - size / 2
                x2, y2 = cx + size / 2, cy + size / 2
                face_boxes.append([x1, y1, x2, y2])
        
        if not face_boxes:
            return None
        
        # Union bbox
        boxes = np.array(face_boxes)
        x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
        x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
        return [x1, y1, x2, y2]
    
    def track_frame(self, img_rgb, union_box=None, last_result=None):
        """Track FLAME parameters for one frame."""
        ret_images = {}
        result = {}
        
        # Detect face
        det_info, _ = self.dwpose_detector(img_rgb)
        if det_info['faces'] is None:
            return None, None
        
        # Crop face
        crop_info = crop_image(img_rgb, det_info['faces'], dsize=512, scale=1.75)
        ret_images['head_image'] = crop_info['img_crop']
        result['head_crop'] = {'M_o2c': crop_info['M_o2c'], 'M_c2o': crop_info['M_c2o']}
        
        # Detect landmarks
        lmk203 = self.landmark_runner.run(crop_info['img_crop'])['pts']
        t_img = crop_info['img_crop'].transpose((2, 0, 1)).astype(np.float32)
        lmk70 = self.lmk70_detector.run(t_img[None] / 255.)['pts'] * 2
        lmk_mp = self.mp_detector.run(crop_info['img_crop'])['pts']
        
        if lmk203 is None or lmk_mp is None or lmk70 is None:
            if last_result is not None:
                result.update({
                    'head_lmk_203': last_result['head_lmk_203'],
                    'head_lmk_70': last_result['head_lmk_70'],
                    'head_lmk_mp': last_result['head_lmk_mp']
                })
            else:
                return None, None
        else:
            if len(lmk203.shape) == 3: lmk203 = lmk203[0]
            if len(lmk70.shape) == 3: lmk70 = lmk70[0]
            if len(lmk_mp.shape) == 3: lmk_mp = lmk_mp[0]
            result.update({
                'head_lmk_203': lmk203,
                'head_lmk_70': lmk70,
                'head_lmk_mp': lmk_mp
            })
        
        # Encode FLAME
        cropped_image = cv2.resize(crop_info['img_crop'], (224, 224))
        cropped_image = np.transpose(cropped_image, (2, 0, 1))[None, ...] / 255.0
        coeff_param = self.teaser_encoder(cropped_image.astype(np.float32))
        
        # Convert camera to perspective
        cam_tensor = torch.from_numpy(coeff_param['cam'])
        R, T = cam2persp_cam_fov(cam_tensor, tanfov=1/24)
        coeff_param['camera_RT_params'] = torch.cat((R, T[..., None]), axis=-1).numpy()
        
        result['flame_coeffs'] = coeff_param
        
        return ret_images, result
    
    def execute(self, video_list):
        """Process list of videos."""
        for video_idx, video_fp in enumerate(video_list):
            video_name = self.get_video_name(video_fp)
            saving_root = osp.join(self.cfg.output_dir, video_name)
            
            log(f"\n[{video_idx+1}/{len(video_list)}] Processing: {video_name}")
            
            # Setup paths
            out_lmdb_dir = osp.join(saving_root, 'img_lmdb')
            base_track_fp = osp.join(saving_root, 'base_tracking_flame.pkl')
            optim_track_fp = osp.join(saving_root, 'optim_tracking_flame.pkl')
            id_params_fp = osp.join(saving_root, 'flame_shape_params.pkl')
            videos_info_path = osp.join(saving_root, 'video_info.json')
            skipped_flag = osp.join(saving_root, 'skipped.txt')
            
            os.makedirs(saving_root, exist_ok=True)
            os.makedirs(out_lmdb_dir, exist_ok=True)
            
            # Skip if already processed
            if osp.exists(skipped_flag):
                log("  ⊗ Skipped (flagged)")
                continue
            if osp.exists(optim_track_fp) and not self.cfg.save_vis_video:
                log("  ✓ Already processed")
                continue
            
            # Open video
            reader = imageio.get_reader(video_fp)
            num_frames = reader.count_frames()
            
            # Frame interval
            frame_interval = 1
            if self.cfg.tracking_with_interval:
                frame_interval = self.cfg.default_frame_interval
                if num_frames // frame_interval > self.cfg.max_frames:
                    frame_interval = int(math.ceil(num_frames / self.cfg.max_frames))
                elif num_frames // frame_interval < self.cfg.min_frames:
                    frame_interval = int(num_frames // self.cfg.min_frames)
                if frame_interval < 1:
                    log(f"  ⊗ Too few frames")
                    continue
            
            # Base tracking
            if not osp.exists(base_track_fp):
                log(f"  → Base tracking ({num_frames} frames, interval={frame_interval})")
                
                lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                base_results = {}
                shape_results = []
                
                last_result = None
                valid_frames = 0
                
                with torch.no_grad():
                    for idx in tqdm(range(0, num_frames, frame_interval),
                                  desc='Extracting frames',
                                  total=num_frames // frame_interval):
                        img_rgb = reader.get_data(idx)
                        frame_name = f'frame_{valid_frames:06d}'
                        
                        ret_images, result = self.track_frame(img_rgb, last_result=last_result)
                        last_result = result
                        
                        if result is None:
                            continue
                        
                        # Save to LMDB
                        for k, v in ret_images.items():
                            if len(v.shape) == 2:
                                v = v[:, :, None]
                            lmdb_engine.dump(f'{frame_name}/{k}',
                                           payload=image2tensor(v, norm=False),
                                           type='image')
                        
                        # Store result
                        shape_params = result['flame_coeffs'].pop('shape_params')
                        base_results[frame_name] = result
                        shape_results.append(shape_params)
                        valid_frames += 1
                
                # Check minimum frames
                if valid_frames < self.cfg.min_frames:
                    lmdb_engine.close()
                    log(f"  ⊗ Insufficient valid frames ({valid_frames} < {self.cfg.min_frames})")
                    with open(skipped_flag, 'w') as f:
                        f.write(f"Insufficient valid frames: {valid_frames} < {self.cfg.min_frames}")
                    continue
                
                # Average shape
                mean_shape = np.array(shape_results).mean(0)
                write_dict_pkl(id_params_fp, {'flame_shape': mean_shape})
                write_dict_pkl(base_track_fp, base_results)
                lmdb_engine.close()
                log(f"  ✓ Base tracking complete ({valid_frames} frames)")
            else:
                log("  → Loading base tracking")
                base_results = load_dict_pkl(base_track_fp)
                valid_frames = len(base_results)
            
            # Load shape params
            id_params = load_dict_pkl(id_params_fp)
            
            # FLAME optimization
            optimized_results = base_results
            if not self.cfg.skip_optimization:
                if not osp.exists(optim_track_fp):
                    log(f"  → FLAME optimization ({self.cfg.flame_optim_steps} steps)")
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=True)
                    self.flame_opt.saving_root = saving_root
                    
                    opt_flame_coeffs, id_params = self.flame_opt.run(
                        base_results,
                        id_params,
                        lmdb_engine,
                        frame_interval=frame_interval,
                        steps=self.cfg.flame_optim_steps
                    )
                    
                    for key in base_results.keys():
                        base_results[key]['flame_coeffs'] = opt_flame_coeffs[key]
                    
                    write_dict_pkl(optim_track_fp, base_results)
                    write_dict_pkl(id_params_fp, id_params)
                    lmdb_engine.close()
                    optimized_results = base_results
                    log("  ✓ Optimization complete")
                else:
                    log("  → Loading optimized tracking")
                    optimized_results = load_dict_pkl(optim_track_fp)
            
            # Save video info
            frames_keys = list(optimized_results.keys())
            videos_info = {
                video_name: {
                    "frames_num": len(frames_keys),
                    "frames_keys": frames_keys
                }
            }
            with open(videos_info_path, 'w') as f:
                json.dump(videos_info, f, indent=2)
            
            # Visualization
            if self.cfg.save_vis_video:
                vis_video_fp = osp.join(saving_root, 'viz_tracking.mp4')
                if not osp.exists(vis_video_fp):
                    log("  → Rendering visualization")
                    lmdb_engine = LMDBEngine(out_lmdb_dir, write=False)
                    all_images = []
                    
                    cameras_kwargs = self.flame_opt.build_cameras_kwargs(1)
                    
                    with torch.no_grad():
                        lights = PointLights(device=self.device, location=[[0.0, -1.0, -100.0]])
                        
                        for frame_key in tqdm(sorted(optimized_results.keys()),
                                            desc='Rendering frames'):
                            # Get FLAME params
                            flame_coeffs = {
                                k: torch.from_numpy(v)[None].to(self.device)
                                for k, v in optimized_results[frame_key]['flame_coeffs'].items()
                            }
                            flame_coeffs['shape_params'] = torch.from_numpy(
                                id_params['flame_shape']
                            )[None].to(self.device)
                            
                            # Generate mesh
                            ret = self.flame(flame_coeffs)
                            vertices = ret['vertices']
                            
                            # Build camera from optimized RT
                            camera_RT = flame_coeffs['camera_RT_params']
                            R = camera_RT[:, :3, :3]
                            T = camera_RT[:, :3, 3]
                            cameras = GS_Camera(R=R, T=T, **cameras_kwargs).to(self.device)
                            
                            # Render mesh
                            rendered = self.renderer.render_mesh(
                                vertices, cameras=cameras, lights=lights
                            )
                            # rendered: [1, 4, H, W] (RGB + alpha, 0-255)
                            render_rgb = rendered[0, :3].cpu().numpy().transpose(1, 2, 0)
                            render_rgb = np.clip(render_rgb, 0, 255).astype(np.uint8)
                            render_alpha = rendered[0, 3].cpu().numpy()
                            
                            # Load input head image
                            input_img = lmdb_engine[f'{frame_key}/head_image'].numpy()
                            input_img = input_img.transpose(1, 2, 0)
                            input_img = np.clip(input_img, 0, 255).astype(np.uint8)
                            
                            # Alpha-composite mesh onto input image
                            alpha_3ch = np.stack([render_alpha] * 3, axis=-1) / 255.0
                            overlay = (input_img * (1 - alpha_3ch) + render_rgb * alpha_3ch).astype(np.uint8)
                            
                            # Side by side: [input | mesh | overlay]
                            combined = np.hstack([input_img, render_rgb, overlay])
                            all_images.append(combined)
                    
                    lmdb_engine.close()
                    images2video(all_images, vis_video_fp, fps=30)
                    log(f"  ✓ Visualization saved: {vis_video_fp}")
            
            log(f"  ✓ Done: {saving_root}")
