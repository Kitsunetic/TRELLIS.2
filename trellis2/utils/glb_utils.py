from __future__ import annotations

from typing import Any

import numpy as np
import o_voxel

_X_PLUS_90 = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def to_glb_z_up(*args, **kwargs) -> Any:
    """
    Export a textured GLB while preserving TRELLIS.2's internal Z-up frame.

    The upstream o_voxel exporter applies an X-axis -90 degree conversion for
    glTF/Y-up convenience. This wrapper applies the inverse transform so the
    saved GLB matches the in-memory mesh and renderer orientation used in this
    repo.
    """
    glb = o_voxel.postprocess.to_glb(*args, **kwargs)
    glb.apply_transform(_X_PLUS_90)
    return glb
