"""
Utilities for reading/writing GLB morph targets using pygltflib.
"""

import struct
import numpy as np
from pygltflib import GLTF2, Accessor, BufferView
from pygltflib.validator import summary as gltf_summary


# glTF component type constants
FLOAT = 5126  # GL_FLOAT
UNSIGNED_SHORT = 5123
ARRAY_BUFFER = 34962


def load_glb(path: str) -> GLTF2:
    """Load a GLB file."""
    return GLTF2().load(path)


def save_glb(gltf: GLTF2, path: str):
    """Save a GLTF2 object to a GLB file."""
    gltf.save(path)


def get_accessor_data(gltf: GLTF2, accessor_index: int) -> np.ndarray:
    """Read accessor data as a numpy array of float32 VEC3 values."""
    accessor = gltf.accessors[accessor_index]
    buffer_view = gltf.bufferViews[accessor.bufferView]

    # Get the binary blob
    blob = gltf.binary_blob()

    byte_offset = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)

    if accessor.type == "VEC3":
        components = 3
    elif accessor.type == "VEC4":
        components = 4
    elif accessor.type == "VEC2":
        components = 2
    elif accessor.type == "SCALAR":
        components = 1
    else:
        raise ValueError(f"Unsupported accessor type: {accessor.type}")

    if accessor.componentType == FLOAT:
        dtype = np.float32
        comp_size = 4
    elif accessor.componentType == UNSIGNED_SHORT:
        dtype = np.uint16
        comp_size = 2
    else:
        raise ValueError(f"Unsupported component type: {accessor.componentType}")

    stride = buffer_view.byteStride or (components * comp_size)
    count = accessor.count

    if stride == components * comp_size:
        # Tightly packed — fast path
        length = count * components * comp_size
        raw = blob[byte_offset : byte_offset + length]
        data = np.frombuffer(raw, dtype=dtype).reshape(count, components)
    else:
        # Strided access
        data = np.zeros((count, components), dtype=dtype)
        for i in range(count):
            offset = byte_offset + i * stride
            raw = blob[offset : offset + components * comp_size]
            data[i] = np.frombuffer(raw, dtype=dtype)

    return data.astype(np.float32)


def find_face_mesh(gltf: GLTF2, mesh_name: str = None):
    """
    Find the face/head mesh primitive in the GLB.
    Returns (mesh_index, primitive_index) or None.

    If mesh_name is provided, searches for an exact (case-insensitive) match.
    Otherwise uses heuristics: looks for meshes with names containing
    'head', 'face', or 'Wolf3D_Head', falling back to the mesh with
    the most vertices.
    """
    if mesh_name:
        target = mesh_name.lower()
        for mi, mesh in enumerate(gltf.meshes):
            if (mesh.name or "").lower() == target:
                return (mi, 0)
        # Partial match fallback
        for mi, mesh in enumerate(gltf.meshes):
            if target in (mesh.name or "").lower():
                return (mi, 0)
        return None

    best = None
    best_vcount = 0
    face_candidate = None
    face_vcount = 0

    for mi, mesh in enumerate(gltf.meshes):
        name = (mesh.name or "").lower()
        for pi, prim in enumerate(mesh.primitives):
            if prim.attributes.POSITION is None:
                continue
            acc = gltf.accessors[prim.attributes.POSITION]
            vcount = acc.count

            # Prefer meshes with face-related names
            if any(kw in name for kw in ["head", "face", "wolf3d_head"]):
                if face_candidate is None or vcount > face_vcount:
                    face_candidate = (mi, pi)
                    face_vcount = vcount

            if vcount > best_vcount:
                best = (mi, pi)
                best_vcount = vcount

    return face_candidate or best


def get_existing_morph_target_names(gltf: GLTF2, mesh_index: int) -> list:
    """Get the list of morph target names from mesh extras or targetNames."""
    mesh = gltf.meshes[mesh_index]

    # Check mesh.extras.targetNames (common convention)
    if mesh.extras and isinstance(mesh.extras, dict):
        names = mesh.extras.get("targetNames", [])
        if names:
            return names

    return []


def get_morph_target_data(gltf: GLTF2, mesh_index: int, prim_index: int):
    """
    Extract existing morph target displacement data.
    Returns dict of {name: np.ndarray of shape (V, 3)}.
    """
    mesh = gltf.meshes[mesh_index]
    prim = mesh.primitives[prim_index]
    names = get_existing_morph_target_names(gltf, mesh_index)

    targets = {}
    if prim.targets:
        for i, target in enumerate(prim.targets):
            name = names[i] if i < len(names) else f"target_{i}"
            if target.POSITION is not None:
                targets[name] = get_accessor_data(gltf, target.POSITION)

    return targets
