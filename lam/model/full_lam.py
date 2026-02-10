"""
Full LAM model: encoder + transformer + GS3DRenderer (same as LAM repo).

Use this for full LAM inference with lam.rendering.GS3DRenderer.
"""

import os
import math
from collections import defaultdict
import torch
import torch.nn as nn
from einops import rearrange, repeat

from .transformer import TransformerDecoder
from ..rendering import GS3DRenderer, PointEmbed

try:
    from diffusers.utils import is_torch_version
except Exception:
    is_torch_version = lambda *a, **k: False


def _encoder_fn(encoder_type: str):
    if encoder_type in ("dino", "dinov2"):
        from ..encoder.dinov2_fusion_wrapper import Dinov2FusionWrapper
        return Dinov2FusionWrapper
    raise ValueError(f"Unknown encoder_type: {encoder_type}")


class FullLAMModel(nn.Module):
    """
    Full LAM: encoder + transformer + GS3DRenderer (same functionality as LAM repo).
    """
    def __init__(
        self,
        transformer_dim: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_type: str = "cond",
        tf_grad_ckpt: bool = False,
        encoder_grad_ckpt: bool = False,
        encoder_freeze: bool = True,
        encoder_type: str = "dino",
        encoder_model_name: str = "dinov2_vits14_reg",
        encoder_feat_dim: int = 384,
        num_pcl: int = 2048,
        pcl_dim: int = 512,
        human_model_path: str = "./model_zoo/human_parametric_models",
        flame_subdivide_num: int = 2,
        flame_type: str = "flame",
        gs_query_dim: int = 64,
        gs_use_rgb: bool = True,
        gs_sh: int = 3,
        gs_mlp_network_config=None,
        gs_xyz_offset_max_step: float = 1.8 / 32,
        gs_clip_scaling: float = 0.2,
        shape_param_dim: int = 100,
        expr_param_dim: int = 50,
        fix_opacity: bool = False,
        fix_rotation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        self.encoder_feat_dim = encoder_feat_dim
        self.conf_use_pred_img = False

        EncoderCls = _encoder_fn(encoder_type)
        self.encoder = EncoderCls(
            model_name=encoder_model_name,
            modulation_dim=None,
            freeze=encoder_freeze,
            encoder_feat_dim=encoder_feat_dim,
        )

        skip_decoder = False
        self.latent_query_points_type = kwargs.get("latent_query_points_type", "e2e_flame")
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl
            self.pcl_embeddings = nn.Embedding(num_pcl, pcl_dim)
        elif self.latent_query_points_type.startswith("flame"):
            latent_query_points_file = os.path.join(
                human_model_path, "flame_points", f"{self.latent_query_points_type}.npy"
            )
            pcl_embeddings = torch.from_numpy(
                __import__("numpy").load(latent_query_points_file)
            ).float()
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith("e2e_flame"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        else:
            raise NotImplementedError(self.latent_query_points_type)

        self.transformer = TransformerDecoder(
            block_type=transformer_type,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            inner_dim=transformer_dim,
            cond_dim=encoder_feat_dim,
            mod_dim=None,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        self.renderer = GS3DRenderer(
            human_model_path=human_model_path,
            subdivide_num=flame_subdivide_num,
            smpl_type=flame_type,
            feat_dim=transformer_dim,
            query_dim=gs_query_dim,
            use_rgb=gs_use_rgb,
            sh_degree=gs_sh,
            mlp_network_config=gs_mlp_network_config,
            xyz_offset_max_step=gs_xyz_offset_max_step,
            clip_scaling=gs_clip_scaling,
            scale_sphere=kwargs.get("scale_sphere", False),
            shape_param_dim=shape_param_dim,
            expr_param_dim=expr_param_dim,
            fix_opacity=fix_opacity,
            fix_rotation=fix_rotation,
            skip_decoder=skip_decoder,
            decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
            gradient_checkpointing=self.gradient_checkpointing,
            add_teeth=kwargs.get("add_teeth", True),
            teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
            oral_mesh_flag=kwargs.get("oral_mesh_flag", False),
            use_mesh_shading=kwargs.get("use_mesh_shading", False),
            render_rgb=kwargs.get("render_rgb", True),
        )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight

    def forward_transformer(
        self, image_feats, camera_embeddings, query_points, query_feats=None
    ):
        B = image_feats.shape[0]
        if self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x = self.pcl_embeddings(range_).unsqueeze(0).repeat(B, 1, 1)
        elif self.latent_query_points_type.startswith("flame"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat(B, 1, 1)
        elif self.latent_query_points_type.startswith("e2e_flame"):
            x = self.pcl_embed(query_points)
        else:
            raise NotImplementedError
        x = x.to(image_feats.dtype)
        if query_feats is not None:
            x = x + query_feats.to(image_feats.dtype)
        x = self.transformer(x, cond=image_feats, mod=camera_embeddings)
        return x

    def forward_encode_image(self, image):
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder), image, **ckpt_kwargs
            )
        else:
            image_feats = self.encoder(image)
        return image_feats

    def forward_latent_points(
        self, image, camera, query_points=None, additional_features=None
    ):
        B = image.shape[0]
        image_feats = self.forward_encode_image(image)
        assert image_feats.shape[-1] == self.encoder_feat_dim
        query_feats = None
        if additional_features is not None and len(additional_features.keys()) > 0:
            if hasattr(self.renderer, "get_batch_project_feats"):
                image_feats_bchw = rearrange(
                    image_feats, "b (h w) c -> b c h w",
                    h=int(math.sqrt(image_feats.shape[1])),
                )
                additional_features["source_image_feats"] = image_feats_bchw
                proj_feats = self.renderer.get_batch_project_feats(
                    None, query_points,
                    additional_features=additional_features,
                    feat_nms=["source_image_feats"],
                    use_mesh=True,
                )
                query_feats = proj_feats["source_image_feats"]
        tokens = self.forward_transformer(
            image_feats, camera_embeddings=None,
            query_points=query_points, query_feats=query_feats
        )
        return tokens, image_feats

    def forward(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        flame_params,
        source_flame_params=None,
        render_images=None,
        data=None,
    ):
        assert image.shape[0] == render_c2ws.shape[0]
        assert image.shape[0] == render_bg_colors.shape[0]
        assert image.shape[0] == flame_params["betas"].shape[0]
        assert image.shape[0] == flame_params["expr"].shape[0]
        render_h = int(render_intrs[0, 0, 1, 2] * 2)
        render_w = int(render_intrs[0, 0, 0, 2] * 2)
        query_points = None

        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params = self.renderer.get_query_points(
                flame_params, device=image.device
            )

        additional_features = {}
        latent_points, image_feats = self.forward_latent_points(
            image[:, 0], camera=None,
            query_points=query_points,
            additional_features=additional_features,
        )
        additional_features.update({
            "image_feats": image_feats,
            "image": image[:, 0],
        })
        image_feats_bchw = rearrange(
            image_feats, "b (h w) c -> b c h w",
            h=int(math.sqrt(image_feats.shape[1])),
        )
        additional_features["image_feats_bchw"] = image_feats_bchw

        render_results = self.renderer(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=flame_params,
            c2w=render_c2ws,
            intrinsic=render_intrs,
            height=render_h,
            width=render_w,
            background_color=render_bg_colors,
            additional_features=additional_features,
        )
        return {"latent_points": latent_points, **render_results}

    @torch.no_grad()
    def infer_single_view(
        self,
        image,
        source_c2ws,
        source_intrs,
        render_c2ws,
        render_intrs,
        render_bg_colors,
        flame_params,
    ):
        assert image.shape[0] == 1
        num_views = render_c2ws.shape[1]
        render_h = int(render_intrs[0, 0, 1, 2] * 2)
        render_w = int(render_intrs[0, 0, 0, 2] * 2)
        query_points = None

        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params = self.renderer.get_query_points(
                flame_params, device=image.device
            )
        latent_points, image_feats = self.forward_latent_points(
            image[:, 0], camera=None, query_points=query_points
        )
        image_feats_bchw = rearrange(
            image_feats, "b (h w) c -> b c h w",
            h=int(math.sqrt(image_feats.shape[1])),
        )
        gs_model_list, query_points, flame_params, _ = self.renderer.forward_gs(
            gs_hidden_features=latent_points,
            query_points=query_points,
            flame_data=flame_params,
            additional_features={
                "image_feats": image_feats,
                "image": image[:, 0],
                "image_feats_bchw": image_feats_bchw,
            },
        )

        render_res_list = []
        for view_idx in range(num_views):
            render_res = self.renderer.forward_animate_gs(
                gs_model_list,
                query_points,
                self.renderer.get_single_view_smpl_data(flame_params, view_idx),
                render_c2ws[:, view_idx : view_idx + 1],
                render_intrs[:, view_idx : view_idx + 1],
                render_h,
                render_w,
                render_bg_colors[:, view_idx : view_idx + 1],
            )
            render_res_list.append(render_res)

        out = defaultdict(list)
        for res in render_res_list:
            for k, v in res.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.concat(v, dim=1)
                if k in ["comp_rgb", "comp_mask", "comp_depth"]:
                    out[k] = out[k][0].permute(0, 2, 3, 1)
            else:
                out[k] = v
        out["cano_gs_lst"] = gs_model_list
        return out
