"""
Inject morph targets (blendshapes) into a GLB file.

This module handles the low-level glTF binary manipulation needed to
add new morph targets to an existing mesh primitive.
"""

import struct
import numpy as np
from pygltflib import (
    GLTF2,
    Accessor,
    BufferView,
    Primitive,
)

FLOAT = 5126
ARRAY_BUFFER = 34962


def _pack_vec3_array(data: np.ndarray) -> bytes:
    """Pack an (N, 3) float32 array into raw bytes."""
    return data.astype(np.float32).tobytes()


def _compute_bounds(data: np.ndarray):
    """Compute min/max for an (N, 3) array."""
    mins = data.min(axis=0).tolist()
    maxs = data.max(axis=0).tolist()
    return mins, maxs


def inject_morph_targets(
    gltf: GLTF2,
    mesh_index: int,
    prim_index: int,
    blendshapes: dict,
    blendshape_order: list = None,
) -> GLTF2:
    """
    Inject morph target displacements into a GLB mesh primitive.

    Args:
        gltf: The GLTF2 object to modify
        mesh_index: Index of the mesh to modify
        prim_index: Index of the primitive within the mesh
        blendshapes: dict of {name: (V, 3) displacement array}
        blendshape_order: ordered list of blendshape names to use.
            If None, uses dict key order.

    Returns:
        Modified GLTF2 object
    """
    if blendshape_order is None:
        blendshape_order = list(blendshapes.keys())

    mesh = gltf.meshes[mesh_index]
    prim = mesh.primitives[prim_index]

    # Get current binary blob
    blob = gltf.binary_blob()
    if blob is None:
        blob = b""

    # We'll append all new morph target data to the existing blob
    new_data = bytearray()
    new_targets = []
    target_names = []

    for name in blendshape_order:
        if name not in blendshapes:
            continue

        disp = blendshapes[name]
        raw = _pack_vec3_array(disp)
        mins, maxs = _compute_bounds(disp)

        # Create buffer view
        bv_index = len(gltf.bufferViews)
        bv = BufferView(
            buffer=0,
            byteOffset=len(blob) + len(new_data),
            byteLength=len(raw),
            target=None,  # morph targets don't need a target
        )
        gltf.bufferViews.append(bv)

        # Create accessor
        acc_index = len(gltf.accessors)
        acc = Accessor(
            bufferView=bv_index,
            byteOffset=0,
            componentType=FLOAT,
            count=disp.shape[0],
            type="VEC3",
            max=maxs,
            min=mins,
        )
        gltf.accessors.append(acc)

        # Build the target dict — pygltflib uses attribute-style access
        from pygltflib import Attributes
        target = Attributes(POSITION=acc_index)
        new_targets.append(target)
        target_names.append(name)

        new_data.extend(raw)

    # Set the targets on the primitive
    # If there were existing targets, we replace them entirely
    prim.targets = new_targets

    # Set default weights (all zero)
    mesh.weights = [0.0] * len(new_targets)

    # Store target names in mesh extras
    if mesh.extras is None:
        mesh.extras = {}
    mesh.extras["targetNames"] = target_names

    # Update the buffer size and binary blob
    combined = bytearray(blob) + new_data
    gltf.buffers[0].byteLength = len(combined)

    # Set the binary blob
    gltf.set_binary_blob(bytes(combined))

    return gltf
