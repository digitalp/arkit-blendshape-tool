"""
Fix skeleton to be TalkingHead-compatible.

Adds missing bones (LeftEye, RightEye, HeadTop_End) that TalkingHead
requires but Avaturn T1 avatars don't have.
"""

import numpy as np
from pygltflib import GLTF2, Node


# Approximate eye positions relative to the Head bone (in local space).
# These are reasonable defaults for a humanoid avatar.
# TalkingHead uses them for eye-contact direction and avatar height estimation.
LEFT_EYE_OFFSET = [0.03, 0.07, 0.08]    # slightly left, up, forward
RIGHT_EYE_OFFSET = [-0.03, 0.07, 0.08]  # slightly right, up, forward
HEAD_TOP_OFFSET = [0.0, 0.15, 0.0]      # straight up from head


def add_missing_bones(gltf: GLTF2) -> tuple:
    """
    Add LeftEye, RightEye, and HeadTop_End bones if missing.
    These are required by TalkingHead for eye tracking and height estimation.

    Also adds them to the skin's joint list so they're part of the skeleton.

    Returns:
        (gltf, added_bones) where added_bones is a list of bone names added
    """
    node_names = {node.name: i for i, node in enumerate(gltf.nodes) if node.name}
    added = []

    # Find the Head bone
    head_idx = node_names.get("Head")
    if head_idx is None:
        print("  WARNING: No 'Head' bone found — cannot add eye bones")
        return gltf, added

    head_node = gltf.nodes[head_idx]
    if head_node.children is None:
        head_node.children = []

    # Add LeftEye if missing
    if "LeftEye" not in node_names:
        eye_node = Node(
            name="LeftEye",
            translation=LEFT_EYE_OFFSET,
            children=[],
        )
        eye_idx = len(gltf.nodes)
        gltf.nodes.append(eye_node)
        head_node.children.append(eye_idx)
        node_names["LeftEye"] = eye_idx
        added.append("LeftEye")

        # Add to skin joints
        _add_to_skin_joints(gltf, eye_idx)

    # Add RightEye if missing
    if "RightEye" not in node_names:
        eye_node = Node(
            name="RightEye",
            translation=RIGHT_EYE_OFFSET,
            children=[],
        )
        eye_idx = len(gltf.nodes)
        gltf.nodes.append(eye_node)
        head_node.children.append(eye_idx)
        node_names["RightEye"] = eye_idx
        added.append("RightEye")

        _add_to_skin_joints(gltf, eye_idx)

    # Add HeadTop_End if missing
    if "HeadTop_End" not in node_names:
        top_node = Node(
            name="HeadTop_End",
            translation=HEAD_TOP_OFFSET,
            children=[],
        )
        top_idx = len(gltf.nodes)
        gltf.nodes.append(top_node)
        head_node.children.append(top_idx)
        node_names["HeadTop_End"] = top_idx
        added.append("HeadTop_End")

        _add_to_skin_joints(gltf, top_idx)

    return gltf, added


def _add_to_skin_joints(gltf: GLTF2, node_idx: int):
    """Add a node index to all skins' joint lists."""
    for skin in gltf.skins:
        if node_idx not in skin.joints:
            skin.joints.append(node_idx)

            # We also need to extend the inverse bind matrices accessor
            # with an identity matrix for the new joint
            if skin.inverseBindMatrices is not None:
                _extend_ibm(gltf, skin.inverseBindMatrices)


def _extend_ibm(gltf: GLTF2, ibm_accessor_idx: int):
    """
    Extend the inverse bind matrices accessor by one identity matrix.
    This is needed when adding a new joint to the skin.
    """
    accessor = gltf.accessors[ibm_accessor_idx]
    bv = gltf.bufferViews[accessor.bufferView]

    # Identity matrix as 16 floats (column-major, as glTF uses)
    identity = np.eye(4, dtype=np.float32).T.tobytes()  # column-major

    blob = bytearray(gltf.binary_blob())

    # Insert the identity matrix at the end of this buffer view's data
    insert_pos = (bv.byteOffset or 0) + bv.byteLength
    blob[insert_pos:insert_pos] = identity

    # Update buffer view length
    bv.byteLength += len(identity)

    # Update accessor count
    accessor.count += 1

    # Shift all buffer views that come after this insertion point
    for other_bv in gltf.bufferViews:
        if other_bv is not bv:
            if (other_bv.byteOffset or 0) >= insert_pos:
                other_bv.byteOffset = (other_bv.byteOffset or 0) + len(identity)

    # Update buffer total size
    gltf.buffers[0].byteLength = len(blob)
    gltf.set_binary_blob(bytes(blob))
