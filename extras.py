"""
Synthesize extra blendshapes that TalkingHead needs from ARKit blendshapes.

TalkingHead expects these 5 additional morph targets beyond the standard
52 ARKit + 15 Oculus visemes:
  - mouthOpen:    derived from jawOpen
  - mouthSmile:   average of mouthSmileLeft + mouthSmileRight
  - eyesClosed:   average of eyeBlinkLeft + eyeBlinkRight
  - eyesLookUp:   average of eyeLookUpLeft + eyeLookUpRight
  - eyesLookDown: average of eyeLookDownLeft + eyeLookDownRight
"""

import numpy as np


def synthesize_extras(blendshapes: dict) -> dict:
    """
    Generate the 5 extra blendshapes from existing ARKit blendshapes.
    Only creates them if they don't already exist in the input dict.

    Args:
        blendshapes: dict of {name: (V, 3) displacement array}

    Returns:
        dict of newly synthesized blendshapes (may be empty if all exist)
    """
    extras = {}

    # Get vertex count from any existing blendshape
    if not blendshapes:
        return extras
    sample = next(iter(blendshapes.values()))
    V = sample.shape[0]
    zero = np.zeros((V, 3), dtype=np.float32)

    # mouthOpen: same as jawOpen
    if "mouthOpen" not in blendshapes:
        extras["mouthOpen"] = blendshapes.get("jawOpen", zero).copy()

    # mouthSmile: average of left + right smile
    if "mouthSmile" not in blendshapes:
        left = blendshapes.get("mouthSmileLeft", zero)
        right = blendshapes.get("mouthSmileRight", zero)
        extras["mouthSmile"] = ((left + right) * 0.5).astype(np.float32)

    # eyesClosed: average of left + right blink
    if "eyesClosed" not in blendshapes:
        left = blendshapes.get("eyeBlinkLeft", zero)
        right = blendshapes.get("eyeBlinkRight", zero)
        extras["eyesClosed"] = ((left + right) * 0.5).astype(np.float32)

    # eyesLookUp: average of left + right look up
    if "eyesLookUp" not in blendshapes:
        left = blendshapes.get("eyeLookUpLeft", zero)
        right = blendshapes.get("eyeLookUpRight", zero)
        extras["eyesLookUp"] = ((left + right) * 0.5).astype(np.float32)

    # eyesLookDown: average of left + right look down
    if "eyesLookDown" not in blendshapes:
        left = blendshapes.get("eyeLookDownLeft", zero)
        right = blendshapes.get("eyeLookDownRight", zero)
        extras["eyesLookDown"] = ((left + right) * 0.5).astype(np.float32)

    return extras
