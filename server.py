#!/usr/bin/env python3
"""
FastAPI server for the ARKit Blendshape + Oculus Viseme tool.
Serves the web UI and handles GLB processing.
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="ARKit Blendshape Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/inspect")
async def inspect_glb(file: UploadFile = File(...)):
    """Inspect a GLB file and return mesh/blendshape info."""
    tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        gltf = load_glb(tmp.name)

        # Validate skeleton
        skel = validate_skeleton(gltf)

        meshes = []
        for mi, mesh in enumerate(gltf.meshes):
            names = get_existing_morph_target_names(gltf, mi)
            prims = []
            for pi, prim in enumerate(mesh.primitives):
                vcount = 0
                if prim.attributes.POSITION is not None:
                    vcount = gltf.accessors[prim.attributes.POSITION].count
                n_targets = len(prim.targets) if prim.targets else 0
                prims.append({
                    "index": pi,
                    "vertexCount": vcount,
                    "morphTargetCount": n_targets,
                })

            arkit = [n for n in names if n in ARKIT_BLENDSHAPES]
            visemes = [n for n in names if n in OCULUS_VISEMES]
            other = [n for n in names
                     if n not in ARKIT_BLENDSHAPES and n not in OCULUS_VISEMES]

            meshes.append({
                "index": mi,
                "name": mesh.name or "(unnamed)",
                "primitives": prims,
                "targetNames": names,
                "arkitCount": len(arkit),
                "visemeCount": len(visemes),
                "otherTargets": other,
            })

        face = find_face_mesh(gltf)
        detected_face = None
        if face:
            mi, pi = face
            detected_face = {
                "meshIndex": mi,
                "primIndex": pi,
                "meshName": gltf.meshes[mi].name or "(unnamed)",
            }

        all_face = find_all_face_meshes(gltf)

        return {
            "meshCount": len(gltf.meshes),
            "nodeCount": len(gltf.nodes),
            "meshes": meshes,
            "detectedFace": detected_face,
            "allFaceMeshes": [
                {"meshIndex": mi, "primIndex": pi,
                 "meshName": gltf.meshes[mi].name}
                for mi, pi in all_face
            ],
            "skeleton": skel,
        }
    finally:
        os.unlink(tmp.name)


@app.post("/api/transfer")
async def transfer_blendshapes(
    reference: UploadFile = File(...),
    target: UploadFile = File(...),
    max_distance: float = Form(0.15),
    falloff_distance: float = Form(0.08),
    face_mesh_ref: str = Form(None),
    face_mesh_target: str = Form(None),
):
    """Transfer blendshapes from reference to target and return the result GLB."""
    ref_tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    tgt_tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
    out_tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)

    try:
        ref_content = await reference.read()
        ref_tmp.write(ref_content)
        ref_tmp.close()

        tgt_content = await target.read()
        tgt_tmp.write(tgt_content)
        tgt_tmp.close()

        ref_gltf = load_glb(ref_tmp.name)
        target_gltf = load_glb(tgt_tmp.name)

        # Find reference face mesh (the one with morph targets)
        ref_mesh_name = face_mesh_ref if face_mesh_ref else None
        ref_face = find_face_mesh(ref_gltf, ref_mesh_name)
        if ref_face is None:
            names = [m.name for m in ref_gltf.meshes]
            raise HTTPException(
                400,
                f"Could not find face mesh in reference. Available: {names}"
            )

        ref_mi, ref_pi = ref_face
        ref_prim = ref_gltf.meshes[ref_mi].primitives[ref_pi]
        ref_positions = get_accessor_data(ref_gltf, ref_prim.attributes.POSITION)
        ref_blendshapes = get_morph_target_data(ref_gltf, ref_mi, ref_pi)

        if not ref_blendshapes:
            raise HTTPException(
                400, "Reference GLB has no morph targets on the face mesh."
            )

        # Find ALL face-related meshes in target (head, eyes, teeth)
        target_face_meshes = find_all_face_meshes(target_gltf)
        if not target_face_meshes:
            # Fallback to single detected face mesh
            tgt_mesh_name = face_mesh_target if face_mesh_target else None
            target_face = find_face_mesh(target_gltf, tgt_mesh_name)
            if target_face is None:
                names = [m.name for m in target_gltf.meshes]
                raise HTTPException(
                    400,
                    f"Could not find face mesh in target. Available: {names}"
                )
            target_face_meshes = [target_face]

        # Build the canonical blendshape order
        order = _build_blendshape_order(ref_blendshapes)

        # Transfer and inject onto each face-related mesh
        for tgt_mi, tgt_pi in target_face_meshes:
            tgt_prim = target_gltf.meshes[tgt_mi].primitives[tgt_pi]
            tgt_positions = get_accessor_data(
                target_gltf, tgt_prim.attributes.POSITION
            )

            transferred = transfer_all_blendshapes(
                ref_positions, tgt_positions, ref_blendshapes,
                max_distance=max_distance,
                falloff_distance=falloff_distance,
            )

            # Synthesize the 5 extra TalkingHead blendshapes
            extras = synthesize_extras(transferred)
            transferred.update(extras)

            mesh_name = target_gltf.meshes[tgt_mi].name or "(unnamed)"
            print(f"  Injecting {len(order)} targets into '{mesh_name}'")

            target_gltf = inject_morph_targets(
                target_gltf, tgt_mi, tgt_pi, transferred, order
            )

        save_glb(target_gltf, out_tmp.name)
        out_bytes = open(out_tmp.name, "rb").read()

        return StreamingResponse(
            io.BytesIO(out_bytes),
            media_type="model/gltf-binary",
            headers={
                "Content-Disposition":
                    "attachment; filename=output_with_blendshapes.glb"
            },
        )
    finally:
        for f in [ref_tmp.name, tgt_tmp.name, out_tmp.name]:
            try:
                os.unlink(f)
            except OSError:
                pass


def _build_blendshape_order(ref_blendshapes: dict) -> list:
    """Build canonical order: mouthOpen first, then visemes, ARKit, extras."""
    order = []
    # TalkingHead reference order: mouthOpen first, then visemes,
    # then mouthSmile, then ARKit, then remaining extras
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
    # Any remaining from reference
    for name in ref_blendshapes:
        if name not in order:
            order.append(name)
    return order


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
