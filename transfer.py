"""
Blendshape transfer using nearest-vertex correspondence.

Given a reference mesh (with blendshapes) and a target mesh (without),
this module transfers the morph target displacements by:
1. Building a KD-tree of the reference mesh vertices
2. For each target vertex, finding the nearest reference vertex
3. Scaling the displacement based on local geometry differences
"""

import numpy as np
from scipy.spatial import cKDTree


def build_correspondence(
    ref_positions: np.ndarray,
    target_positions: np.ndarray,
    max_distance: float = 0.1,
) -> tuple:
    """
    Build vertex correspondence between reference and target meshes
    using nearest-neighbor lookup.

    Args:
        ref_positions: (N, 3) reference mesh vertices
        target_positions: (M, 3) target mesh vertices
        max_distance: max distance threshold for valid correspondence

    Returns:
        (indices, distances, mask):
            indices: (M,) index into ref_positions for each target vertex
            distances: (M,) distance to nearest ref vertex
            mask: (M,) bool, True if correspondence is valid
    """
    tree = cKDTree(ref_positions)
    distances, indices = tree.query(target_positions, k=1)
    mask = distances < max_distance
    return indices, distances, mask


def compute_local_scale(
    ref_positions: np.ndarray,
    target_positions: np.ndarray,
    indices: np.ndarray,
    neighborhood_k: int = 6,
) -> np.ndarray:
    """
    Compute per-vertex scale factor based on local neighborhood size
    differences between reference and target meshes.

    This helps adapt displacements when meshes have different scales
    or proportions in different regions.

    Returns:
        (M,) array of scale factors for each target vertex
    """
    ref_tree = cKDTree(ref_positions)
    target_tree = cKDTree(target_positions)

    # For each target vertex, measure local neighborhood radius
    target_dists, _ = target_tree.query(target_positions, k=neighborhood_k + 1)
    target_radius = np.mean(target_dists[:, 1:], axis=1)  # skip self

    # For corresponding ref vertices, measure local neighborhood radius
    ref_subset = ref_positions[indices]
    ref_dists, _ = ref_tree.query(ref_subset, k=neighborhood_k + 1)
    ref_radius = np.mean(ref_dists[:, 1:], axis=1)

    # Scale factor: target_local_size / ref_local_size
    scale = np.where(ref_radius > 1e-8, target_radius / ref_radius, 1.0)

    # Clamp to reasonable range
    scale = np.clip(scale, 0.5, 2.0)

    return scale


def transfer_blendshape(
    ref_positions: np.ndarray,
    target_positions: np.ndarray,
    ref_displacement: np.ndarray,
    indices: np.ndarray,
    mask: np.ndarray,
    local_scale: np.ndarray,
    falloff_distance: float = 0.05,
    distances: np.ndarray = None,
) -> np.ndarray:
    """
    Transfer a single blendshape displacement from reference to target.

    Args:
        ref_positions: (N, 3) reference neutral vertices
        target_positions: (M, 3) target neutral vertices
        ref_displacement: (N, 3) displacement for this blendshape on ref
        indices: (M,) nearest ref vertex index per target vertex
        mask: (M,) valid correspondence mask
        local_scale: (M,) per-vertex scale factors
        falloff_distance: distance over which displacement fades out
        distances: (M,) distance to nearest ref vertex

    Returns:
        (M, 3) displacement array for the target mesh
    """
    M = target_positions.shape[0]
    target_disp = np.zeros((M, 3), dtype=np.float32)

    # Look up the reference displacement for each target vertex
    mapped_disp = ref_displacement[indices]

    # Apply local scale
    scaled_disp = mapped_disp * local_scale[:, np.newaxis]

    # Apply distance-based falloff for vertices far from correspondence
    if distances is not None:
        weight = np.clip(1.0 - distances / falloff_distance, 0.0, 1.0)
        weight = weight ** 2  # smooth falloff
        scaled_disp *= weight[:, np.newaxis]

    # Apply mask
    target_disp[mask] = scaled_disp[mask]

    return target_disp


def transfer_all_blendshapes(
    ref_positions: np.ndarray,
    target_positions: np.ndarray,
    ref_blendshapes: dict,
    max_distance: float = 0.15,
    falloff_distance: float = 0.08,
) -> dict:
    """
    Transfer all blendshapes from reference to target mesh.

    Args:
        ref_positions: (N, 3) reference neutral vertices
        target_positions: (M, 3) target neutral vertices
        ref_blendshapes: dict of {name: (N, 3) displacement}
        max_distance: max correspondence distance
        falloff_distance: falloff distance for weight blending

    Returns:
        dict of {name: (M, 3) displacement} for target mesh
    """
    print(f"  Building correspondence: {ref_positions.shape[0]} ref -> "
          f"{target_positions.shape[0]} target vertices")

    indices, distances, mask = build_correspondence(
        ref_positions, target_positions, max_distance
    )

    valid_pct = np.sum(mask) / len(mask) * 100
    print(f"  Valid correspondences: {np.sum(mask)}/{len(mask)} ({valid_pct:.1f}%)")

    if valid_pct < 30:
        print("  WARNING: Low correspondence rate. Meshes may be misaligned or "
              "have very different topology. Try increasing --max-distance.")

    local_scale = compute_local_scale(
        ref_positions, target_positions, indices
    )

    result = {}
    for name, ref_disp in ref_blendshapes.items():
        target_disp = transfer_blendshape(
            ref_positions,
            target_positions,
            ref_disp,
            indices,
            mask,
            local_scale,
            falloff_distance,
            distances,
        )
        result[name] = target_disp

    return result
