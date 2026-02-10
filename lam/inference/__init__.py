"""
LAM inference utilities.

- run_lam_repo: run full LAM inference (image + motion_seqs_dir) using the LAM repo.
- Pipeline-based: use lam.pipeline.AnimationPipeline with LAMModel for image + motion .pt.
"""

from .run_lam_repo import run_lam_repo_inference

__all__ = ["run_lam_repo_inference"]
