#!/usr/bin/env python3
"""
ARKit Blendshape + Oculus Viseme Injector for GLB Avatars

Transfers ARKit 52 blendshapes, 15 Oculus visemes, and 5 TalkingHead
extras from a reference GLB onto a target GLB.

Usage:
    python main.py transfer --reference ref.glb --target target.glb --output output.glb
    python main.py inspect --input model.glb
"""

import argparse
import sys

import numpy as np

from blendshape_names import (
    ARKIT_BLENDSHAPES, OCULUS_VISEMES, TALKINGHEAD_EXTRAS, ALL_BLENDSHAPES,
)
from glb_utils import (
    load_glb,
    save_glb,
    find_face_mesh,
    find_all_face_meshes,
    get_accessor_data,
    get_existing_morph_target_names,
    get_morph_target_data,
    validate_skeleton,
)
from transfer import transfer_all_blendshapes
from inject import inject_morph_targets
from extras import synthesize_extras


def cmd_inspect(args):
    """Inspect a GLB file for mesh info and existing morph targets."""
    gltf = load_glb(args.input)

    print(f"\nFile: {args.input}")
    print(f"Meshes: {len(gltf.meshes)}")
    print(f"Nodes: {len(gltf.nodes)}")

    # Skeleton validation
    skel = validate_skeleton(gltf)
    if skel["valid"]:
        print(f"Skeleton: OK ({len(skel['present'])} required bones found)")
    else:
        print(f"Skeleton: INCOMPLETE — missing {len(skel['missing'])} bones:")
        for b in skel["missing"]:
            print(f"    - {b}")

    # Face meshes
    all_face = find_all_face_meshes(gltf)
    if all_face:
        print(f"Face meshes: {len(all_face)}")
        for mi, pi in all_face:
            print(f"    {gltf.meshes[mi].name}")

    print()
    for mi, mesh in enumerate(gltf.meshes):
        print(f"  Mesh {mi}: '{mesh.name or '(unnamed)'}'")
        names = get_existing_morph_target_names(gltf, mi)

        for pi, prim in enumerate(mesh.primitives):
            vcount = 0
            if prim.attributes.POSITION is not None:
                vcount = gltf.accessors[prim.attributes.POSITION].count

            n_targets = len(prim.targets) if prim.targets else 0
            print(f"    Primitive {pi}: {vcount} vertices, "
                  f"{n_targets} morph targets")

            if names:
                arkit = [n for n in names if n in ARKIT_BLENDSHAPES]
                visemes = [n for n in names if n in OCULUS_VISEMES]
                extras = [n for n in names if n in TALKINGHEAD_EXTRAS]
                other = [n for n in names if n not in ALL_BLENDSHAPES]

                print(f"      ARKit: {len(arkit)}/52, "
                      f"Visemes: {len(visemes)}/15, "
                      f"Extras: {len(extras)}/5")
                if other:
                    print(f"      Other: {', '.join(other)}")

    face = find_face_mesh(gltf)
    if face:
        mi, pi = face
        print(f"\n  Primary face mesh: Mesh {mi} "
              f"('{gltf.meshes[mi].name or '(unnamed)'}')")


def cmd_transfer(args):
    """Transfer blendshapes from reference to target GLB."""
    print(f"Loading reference: {args.reference}")
    ref_gltf = load_glb(args.reference)

    print(f"Loading target: {args.target}")
    target_gltf = load_glb(args.target)

    # Validate target skeleton
    skel = validate_skeleton(target_gltf)
    if not skel["valid"]:
        print(f"\nWARNING: Target is missing {len(skel['missing'])} bones "
              f"required by TalkingHead:")
        for b in skel["missing"]:
            print(f"    - {b}")
        print("The output may not work in TalkingHead.\n")

    # Find reference face mesh
    ref_face = find_face_mesh(ref_gltf, args.face_mesh_ref)
    if ref_face is None:
        print("ERROR: Could not find face mesh in reference GLB.")
        print("Available meshes:")
        for i, m in enumerate(ref_gltf.meshes):
            print(f"  {i}: '{m.name}'")
        sys.exit(1)

    ref_mi, ref_pi = ref_face
    print(f"\nReference face: '{ref_gltf.meshes[ref_mi].name}'")

    ref_prim = ref_gltf.meshes[ref_mi].primitives[ref_pi]
    ref_positions = get_accessor_data(ref_gltf, ref_prim.attributes.POSITION)
    ref_blendshapes = get_morph_target_data(ref_gltf, ref_mi, ref_pi)

    if not ref_blendshapes:
        print("ERROR: Reference GLB has no morph targets on the face mesh.")
        sys.exit(1)

    print(f"Reference: {ref_positions.shape[0]} verts, "
          f"{len(ref_blendshapes)} blendshapes")

    # Find ALL face-related meshes in target
    target_face_meshes = find_all_face_meshes(target_gltf)
    if not target_face_meshes:
        target_face = find_face_mesh(target_gltf, args.face_mesh_target)
        if target_face is None:
            print("ERROR: Could not find face mesh in target GLB.")
            for i, m in enumerate(target_gltf.meshes):
                print(f"  {i}: '{m.name}'")
            sys.exit(1)
        target_face_meshes = [target_face]

    print(f"Target face meshes: {len(target_face_meshes)}")
    for mi, pi in target_face_meshes:
        print(f"  '{target_gltf.meshes[mi].name}'")

    # Build canonical blendshape order
    order = []
    if "mouthOpen" in ref_blendshapes:
        order.append("mouthOpen")
    for name in OCULUS_VISEMES:
        if name not in order:
            order.append(name)
    if "mouthSmile" not in order:
        order.append("mouthSmile")
    for name in ARKIT_BLENDSHAPES:
        if name not in order:
            order.append(name)
    for name in TALKINGHEAD_EXTRAS:
        if name not in order:
            order.append(name)
    for name in ref_blendshapes:
        if name not in order:
            order.append(name)

    # Transfer and inject onto each face mesh
    for tgt_mi, tgt_pi in target_face_meshes:
        tgt_prim = target_gltf.meshes[tgt_mi].primitives[tgt_pi]
        tgt_positions = get_accessor_data(
            target_gltf, tgt_prim.attributes.POSITION
        )
        mesh_name = target_gltf.meshes[tgt_mi].name or "(unnamed)"
        print(f"\nTransferring to '{mesh_name}' ({tgt_positions.shape[0]} verts)...")

        transferred = transfer_all_blendshapes(
            ref_positions, tgt_positions, ref_blendshapes,
            max_distance=args.max_distance,
            falloff_distance=args.falloff_distance,
        )

        # Synthesize extra blendshapes
        extras = synthesize_extras(transferred)
        transferred.update(extras)

        print(f"  Injecting {len(order)} morph targets...")
        target_gltf = inject_morph_targets(
            target_gltf, tgt_mi, tgt_pi, transferred, order
        )

    save_glb(target_gltf, args.output)
    print(f"\nSaved: {args.output}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Add ARKit blendshapes and Oculus visemes to GLB avatars"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    p_inspect = subparsers.add_parser(
        "inspect", help="Inspect a GLB for mesh info and morph targets"
    )
    p_inspect.add_argument("--input", "-i", required=True, help="Input GLB file")

    p_transfer = subparsers.add_parser(
        "transfer",
        help="Transfer blendshapes from a reference GLB to a target GLB",
    )
    p_transfer.add_argument(
        "--reference", "-r", required=True,
        help="Reference GLB with existing ARKit blendshapes",
    )
    p_transfer.add_argument(
        "--target", "-t", required=True,
        help="Target GLB without blendshapes",
    )
    p_transfer.add_argument(
        "--output", "-o", required=True,
        help="Output GLB path",
    )
    p_transfer.add_argument(
        "--max-distance", type=float, default=0.15,
        help="Max distance for vertex correspondence (default: 0.15)",
    )
    p_transfer.add_argument(
        "--falloff-distance", type=float, default=0.08,
        help="Distance falloff for displacement blending (default: 0.08)",
    )
    p_transfer.add_argument(
        "--face-mesh-ref", type=str, default=None,
        help="Name of the face mesh in reference (auto-detected if omitted)",
    )
    p_transfer.add_argument(
        "--face-mesh-target", type=str, default=None,
        help="Name of the face mesh in target (auto-detected if omitted)",
    )

    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "transfer":
        cmd_transfer(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
