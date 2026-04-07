#!/usr/bin/env python3
"""
ARKit Blendshape + Oculus Viseme Injector for GLB Avatars

Transfers ARKit 52 blendshapes and 15 Oculus visemes from a reference
GLB (e.g., Avaturn T2) onto a target GLB (e.g., Avaturn T1) that
lacks them.

Usage:
    python main.py transfer --reference ref.glb --target target.glb --output output.glb
    python main.py inspect --input model.glb
"""

import argparse
import sys
import os

import numpy as np

from blendshape_names import ARKIT_BLENDSHAPES, OCULUS_VISEMES, ALL_BLENDSHAPES
from glb_utils import (
    load_glb,
    save_glb,
    find_face_mesh,
    get_accessor_data,
    get_existing_morph_target_names,
    get_morph_target_data,
)
from transfer import transfer_all_blendshapes
from inject import inject_morph_targets


def cmd_inspect(args):
    """Inspect a GLB file for mesh info and existing morph targets."""
    gltf = load_glb(args.input)

    print(f"\nFile: {args.input}")
    print(f"Meshes: {len(gltf.meshes)}")
    print(f"Nodes: {len(gltf.nodes)}")
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
                # Categorize
                arkit = [n for n in names if n in ARKIT_BLENDSHAPES]
                visemes = [n for n in names if n in OCULUS_VISEMES]
                other = [n for n in names
                         if n not in ARKIT_BLENDSHAPES and n not in OCULUS_VISEMES]

                print(f"      ARKit blendshapes: {len(arkit)}/52")
                print(f"      Oculus visemes: {len(visemes)}/15")
                if other:
                    print(f"      Other: {len(other)} ({', '.join(other[:5])}...)")

    # Identify face mesh
    face = find_face_mesh(gltf)
    if face:
        mi, pi = face
        print(f"\n  Detected face mesh: Mesh {mi}, Primitive {pi} "
              f"('{gltf.meshes[mi].name or '(unnamed)'}')")
    else:
        print("\n  Could not auto-detect face mesh.")


def cmd_transfer(args):
    """Transfer blendshapes from reference to target GLB."""
    print(f"Loading reference: {args.reference}")
    ref_gltf = load_glb(args.reference)

    print(f"Loading target: {args.target}")
    target_gltf = load_glb(args.target)

    # Find face meshes
    ref_face = find_face_mesh(ref_gltf, args.face_mesh_ref)
    if ref_face is None:
        print("ERROR: Could not find face mesh in reference GLB.")
        print("Available meshes:")
        for i, m in enumerate(ref_gltf.meshes):
            print(f"  {i}: '{m.name}'")
        print("Use --face-mesh-ref to specify the mesh name.")
        sys.exit(1)

    target_face = find_face_mesh(target_gltf, args.face_mesh_target)
    if target_face is None:
        print("ERROR: Could not find face mesh in target GLB.")
        print("Available meshes:")
        for i, m in enumerate(target_gltf.meshes):
            print(f"  {i}: '{m.name}'")
        print("Use --face-mesh-target to specify the mesh name.")
        sys.exit(1)

    ref_mi, ref_pi = ref_face
    target_mi, target_pi = target_face

    print(f"\nReference face: Mesh {ref_mi} '{ref_gltf.meshes[ref_mi].name}', "
          f"Primitive {ref_pi}")
    print(f"Target face: Mesh {target_mi} '{target_gltf.meshes[target_mi].name}', "
          f"Primitive {target_pi}")

    # Get reference positions and blendshapes
    ref_prim = ref_gltf.meshes[ref_mi].primitives[ref_pi]
    ref_positions = get_accessor_data(ref_gltf, ref_prim.attributes.POSITION)
    print(f"\nReference vertices: {ref_positions.shape[0]}")

    ref_blendshapes = get_morph_target_data(ref_gltf, ref_mi, ref_pi)
    print(f"Reference blendshapes: {len(ref_blendshapes)}")

    if not ref_blendshapes:
        print("ERROR: Reference GLB has no morph targets on the face mesh.")
        sys.exit(1)

    # Report which blendshapes are available
    available_arkit = [n for n in ref_blendshapes if n in ARKIT_BLENDSHAPES]
    available_visemes = [n for n in ref_blendshapes if n in OCULUS_VISEMES]
    print(f"  ARKit: {len(available_arkit)}/52")
    print(f"  Visemes: {len(available_visemes)}/15")

    missing_arkit = [n for n in ARKIT_BLENDSHAPES if n not in ref_blendshapes]
    missing_visemes = [n for n in OCULUS_VISEMES if n not in ref_blendshapes]
    if missing_arkit:
        print(f"  Missing ARKit: {', '.join(missing_arkit[:10])}"
              f"{'...' if len(missing_arkit) > 10 else ''}")
    if missing_visemes:
        print(f"  Missing visemes: {', '.join(missing_visemes[:10])}"
              f"{'...' if len(missing_visemes) > 10 else ''}")

    # Get target positions
    target_prim = target_gltf.meshes[target_mi].primitives[target_pi]
    target_positions = get_accessor_data(target_gltf, target_prim.attributes.POSITION)
    print(f"\nTarget vertices: {target_positions.shape[0]}")

    # Transfer blendshapes
    print("\nTransferring blendshapes...")
    transferred = transfer_all_blendshapes(
        ref_positions,
        target_positions,
        ref_blendshapes,
        max_distance=args.max_distance,
        falloff_distance=args.falloff_distance,
    )

    print(f"\nTransferred {len(transferred)} blendshapes")

    # Determine injection order: ARKit first, then visemes, then others
    order = []
    for name in ARKIT_BLENDSHAPES:
        if name in transferred:
            order.append(name)
    for name in OCULUS_VISEMES:
        if name in transferred:
            order.append(name)
    for name in transferred:
        if name not in order:
            order.append(name)

    # Inject into target
    print(f"Injecting {len(order)} morph targets into target GLB...")
    target_gltf = inject_morph_targets(
        target_gltf, target_mi, target_pi, transferred, order
    )

    # Save
    output = args.output
    save_glb(target_gltf, output)
    print(f"\nSaved: {output}")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Add ARKit blendshapes and Oculus visemes to GLB avatars"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # inspect
    p_inspect = subparsers.add_parser(
        "inspect", help="Inspect a GLB for mesh info and morph targets"
    )
    p_inspect.add_argument("--input", "-i", required=True, help="Input GLB file")

    # transfer
    p_transfer = subparsers.add_parser(
        "transfer",
        help="Transfer blendshapes from a reference GLB to a target GLB",
    )
    p_transfer.add_argument(
        "--reference", "-r", required=True,
        help="Reference GLB with existing ARKit blendshapes (e.g., Avaturn T2)",
    )
    p_transfer.add_argument(
        "--target", "-t", required=True,
        help="Target GLB without blendshapes (e.g., Avaturn T1)",
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
