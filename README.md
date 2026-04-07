# ARKit Blendshape + Oculus Viseme Injector

Transfers 52 ARKit blendshapes and 15 Oculus visemes from a reference GLB avatar onto a target GLB avatar that lacks them. Includes a web UI with 3D preview and blendshape sliders.

## Quick Start (Ubuntu Server)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/arkit-blendshape-tool.git
cd arkit-blendshape-tool

# Install Python 3.10+ if needed
sudo apt update && sudo apt install -y python3 python3-venv python3-pip

# Create venv and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the web UI
python server.py
```

Open `http://YOUR_SERVER_IP:8000` in your browser.

## Usage

### Web UI

1. Open the app in your browser
2. Drop a reference GLB (with ARKit blendshapes, e.g. Avaturn T2) in the left panel
3. Drop a target GLB (without blendshapes, e.g. Avaturn T1) in the left panel
4. Adjust max distance / falloff if needed
5. Click "Transfer Blendshapes"
6. Preview the result in the 3D viewport, test with sliders
7. Download the output GLB

### CLI

```bash
# Inspect a GLB
python main.py inspect -i avatar.glb

# Transfer blendshapes
python main.py transfer -r reference.glb -t target.glb -o output.glb
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-distance` | 0.15 | Max distance for vertex correspondence |
| `--falloff-distance` | 0.08 | Distance falloff for displacement blending |
| `--face-mesh-ref` | auto | Name of face mesh in reference GLB |
| `--face-mesh-target` | auto | Name of face mesh in target GLB |

## Running with systemd (production)

Create `/etc/systemd/system/blendshape-tool.service`:

```ini
[Unit]
Description=ARKit Blendshape Tool
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/arkit-blendshape-tool
ExecStart=/opt/arkit-blendshape-tool/.venv/bin/python server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable blendshape-tool
sudo systemctl start blendshape-tool
```

## How It Works

1. Loads a reference GLB that already has ARKit blendshapes
2. Loads a target GLB without blendshapes
3. Builds vertex correspondence using KD-tree nearest-neighbor matching
4. Transfers morph target displacements with local scale adaptation
5. Injects the morph targets into the target GLB as glTF morph targets

## Blendshapes

### ARKit (52)
Eye blinks, gaze, squint, wide — jaw open/forward/left/right — mouth smile, frown, funnel, pucker, stretch, roll, shrug, press, dimple — brow down/up/inner — cheek puff/squint — nose sneer — tongue out.

### Oculus Visemes (15)
`viseme_sil`, `viseme_PP`, `viseme_FF`, `viseme_TH`, `viseme_DD`, `viseme_kk`, `viseme_CH`, `viseme_SS`, `viseme_nn`, `viseme_RR`, `viseme_aa`, `viseme_E`, `viseme_I`, `viseme_O`, `viseme_U`

## Requirements

- Python 3.10+
- A reference GLB with existing ARKit blendshapes and/or Oculus visemes
- Both meshes should represent similar face geometry (same avatar platform preferred)
