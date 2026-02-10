# LAM Refactored - Modular Architecture

This is a clean, modular refactoring of the LAM (Large Avatar Model) codebase with clear component boundaries.

## Structure

```
lam/
├── flame/          # FLAME parametric head model
├── gaussian/       # 3D Gaussian splatting (TODO)
├── tracking/       # FLAME parameter tracking (TODO)
├── encoder/        # Image encoders (TODO)
├── model/          # LAM transformer (TODO)
├── io/             # Input/output utilities (TODO)
├── utils/          # Shared utilities
├── pipeline/       # High-level pipelines (TODO)
└── examples/       # Usage examples
```

## Installation

```bash
cd /path/to/talking-head
pip install -e ./lam
```

## Quick Start

### Using FLAME Model

```python
from lam.flame import FlameHead, FlameConfig
import torch

# Configure FLAME
config = FlameConfig(
    flame_model_path="LAM/assets/FLAME2023/flame2023.pkl",
    flame_lmk_embedding_path="LAM/assets/landmark_embedding.npy",
    n_shape=100,
    n_expr=50
)

# Create model
flame = FlameHead(config)

# Generate face mesh
output = flame.forward(
    shape_params=torch. zeros(1, 100),
    expr_params=torch.zeros(1, 50),
    pose_params=torch.zeros(1, 3),
    jaw_params=torch.zeros(1, 3),
)

vertices = output["vertices"]  # (1, 5023, 3)
landmarks = output["landmarks"]  # (1, 68, 3)
```

## Features

✅ **Clean Module Boundaries** - Each component is independent
✅ **Easy Testing** - Test components in isolation
✅ **Simple Imports** - `from lam.flame import FlameHead`
✅ **Well Documented** - Clear docstrings and type hints
✅ **Type Hints** - Full type annotations for better IDE support

## Module Status

- ✅ `utils/` - Math, camera, and config utilities
- ✅ `flame/` - FLAME model and LBS
- 🚧 `gaussian/` - Coming soon
- 🚧 `tracking/` - Coming soon
- 🚧 `encoder/` - Coming soon
- 🚧 `model/` - Coming soon
- 🚧 `pipeline/` - Coming soon

## Differences from Original LAM

### Better Organization
- Flat module structure instead of deep nesting
- Clear separation between library and application code
- Independent versioning of components

### Cleaner Interfaces
```python
# Original LAM
from lam.models.rendering.flame_model.flame import FlameHead

# Refactored
from lam.flame import FlameHead
```

### Easier Testing
Each module can be tested independently:
```python
# Test FLAME alone
pytest tests/test_flame.py

# Test Gaussian rendering alone
pytest tests/test_gaussian.py
```

## Contributing

When adding new components:
1. Create a new module directory
2. Add `__init__.py` with clean exports
3. Implement with clear interfaces
4. Add docstrings and type hints
5. Create tests in `tests/`
6. Add usage examples in `examples/`

## License

Same as original LAM project (Apache 2.0)
