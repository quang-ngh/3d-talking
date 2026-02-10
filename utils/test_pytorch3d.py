import torch
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

# 1. Create a simple mesh (e.g., a sphere)
verts = torch.tensor([[-1., -1., -1.], [1., -1., -1.], [-1., 1., -1.], [1., 1., -1.],
                      [-1., -1., 1.], [1., -1., 1.], [-1., 1., 1.], [1., 1., 1.]], dtype=torch.float32)
faces = torch.tensor([[0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6],
                      [0, 4, 6], [0, 6, 2], [1, 5, 7], [1, 7, 3],
                      [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6]], dtype=torch.int64)
mesh = Meshes(verts=[verts], faces=[faces])

# 2. Sample points from the mesh
sampled_points = sample_points_from_meshes(mesh, num_samples=100)

# 3. Create a "ground truth" or reference set of points (e.g., a slightly perturbed version)
perturbed_points = sampled_points + 0.01 * torch.randn_like(sampled_points)
