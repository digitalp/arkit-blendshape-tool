#!/usr/bin/env bash
#
# Deploy latest changes from git to your Ubuntu server.
#
# Usage:
#   ./deploy.sh user@your-server-ip
#   ./deploy.sh user@your-server-ip /opt/arkit-blendshape-tool
#
# What it does:
#   1. SSHs into the server
#   2. Clones the repo (first run) or pulls latest changes
#   3. Installs any new pip dependencies if requirements.txt changed
#   4. Restarts the systemd service (if it exists) or tells you to restart manually
#

set -euo pipefail

REMOTE="${1:?Usage: ./deploy.sh user@host [remote_path]}"
REMOTE_PATH="${2:-~/arkit-blendshape-tool}"
REPO="https://github.com/digitalp/arkit-blendshape-tool.git"

echo "==> Deploying to ${REMOTE}:${REMOTE_PATH}"

ssh "${REMOTE}" bash -s "${REMOTE_PATH}" "${REPO}" << 'EOF'
set -euo pipefail

REMOTE_PATH="$1"
REPO="$2"

# Clone or pull
if [ -d "${REMOTE_PATH}/.git" ]; then
  echo "Pulling latest changes..."
  cd "${REMOTE_PATH}"
  git fetch origin
  git reset --hard origin/main 2>/dev/null || git reset --hard origin/master
else
  echo "Cloning repo..."
  git clone "${REPO}" "${REMOTE_PATH}"
  cd "${REMOTE_PATH}"
fi

# Set up venv and install deps
echo "Installing dependencies..."
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

# Restart service
echo "Restarting service..."
if systemctl is-active --quiet blendshape-tool 2>/dev/null; then
  sudo systemctl restart blendshape-tool
  echo "systemd service restarted"
else
  echo "No systemd service found. Restart manually:"
  echo "  cd ${REMOTE_PATH} && source .venv/bin/activate && python server.py"
fi

echo "Done"
EOF
