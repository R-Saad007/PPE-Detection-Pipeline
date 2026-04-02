#!/usr/bin/env bash
# setup-api-droplet.sh — Bootstrap the PPE Detection API on a fresh Ubuntu droplet.
#
# Usage (run as root on the new droplet):
#   curl -sSL <raw-url> | bash
#   — or —
#   scp setup-api-droplet.sh root@DROPLET_IP:~ && ssh root@DROPLET_IP bash setup-api-droplet.sh
#
# Prerequisites: Ubuntu 22.04+ droplet, 2 vCPU / 4 GB RAM, SGP1 region.

set -euo pipefail

APP_DIR="/opt/ppe-detection"
APP_USER="ppe"
FTP_SERVER_IP="146.190.109.74"

echo "==> Updating packages..."
apt-get update -qq && apt-get upgrade -y -qq

echo "==> Installing Python 3 and venv..."
apt-get install -y -qq python3 python3-pip python3-venv ufw

# ── Create service user ──────────────────────────────────────────────────
echo "==> Creating service user: ${APP_USER}..."
id -u "${APP_USER}" &>/dev/null || useradd --system --shell /usr/sbin/nologin "${APP_USER}"

# ── Application directory ────────────────────────────────────────────────
echo "==> Setting up ${APP_DIR}..."
mkdir -p "${APP_DIR}"

echo "    Copy your project files into ${APP_DIR} now (git clone or scp)."
echo "    Required: src/, config/, requirements.txt, models/yolov8s.pt, .env"
echo ""

# If the repo is already cloned, set up venv and install deps
if [ -f "${APP_DIR}/requirements.txt" ]; then
    echo "==> Creating virtual environment..."
    python3 -m venv "${APP_DIR}/venv"

    echo "==> Installing Python dependencies..."
    "${APP_DIR}/venv/bin/pip" install --upgrade pip -q
    "${APP_DIR}/venv/bin/pip" install -r "${APP_DIR}/requirements.txt" -q

    echo "==> Creating working directories..."
    mkdir -p "${APP_DIR}"/{uploads,outputs,failed,logs}
    chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
else
    echo "    [SKIP] requirements.txt not found — run this section manually after copying files."
fi

# ── Firewall — port 5000 only from FTP server ───────────────────────────
echo "==> Configuring firewall (UFW)..."
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow from "${FTP_SERVER_IP}" to any port 5000 proto tcp comment "PPE API from FTP server"
ufw --force enable
echo "    Port 5000 open ONLY from ${FTP_SERVER_IP}"

# ── Systemd service ─────────────────────────────────────────────────────
echo "==> Installing systemd service..."
if [ -f "${APP_DIR}/deployment/ppe-detection-api.service" ]; then
    cp "${APP_DIR}/deployment/ppe-detection-api.service" /etc/systemd/system/
    systemctl daemon-reload
    echo "    Service installed. Enable with: systemctl enable --now ppe-detection-api"
else
    echo "    [SKIP] Service file not found — copy it manually later."
fi

# ── HuggingFace model cache directory ────────────────────────────────────
echo "==> Setting up model cache..."
mkdir -p /home/"${APP_USER}"/.cache/huggingface
chown -R "${APP_USER}:${APP_USER}" /home/"${APP_USER}" 2>/dev/null || true

echo ""
echo "============================================================"
echo "  Setup complete. Next steps:"
echo ""
echo "  1. Copy project files to ${APP_DIR}/"
echo "     scp -r src/ config/ requirements.txt models/ .env root@THIS_IP:${APP_DIR}/"
echo ""
echo "  2. If not already done, install deps:"
echo "     python3 -m venv ${APP_DIR}/venv"
echo "     ${APP_DIR}/venv/bin/pip install -r ${APP_DIR}/requirements.txt"
echo ""
echo "  3. Ensure models/yolov8s.pt is in place"
echo "     (PPE model auto-downloads from HuggingFace on first run)"
echo ""
echo "  4. Set ownership and start:"
echo "     chown -R ${APP_USER}:${APP_USER} ${APP_DIR}"
echo "     systemctl enable --now ppe-detection-api"
echo ""
echo "  5. Verify:"
echo "     curl http://localhost:5000/health"
echo "     journalctl -u ppe-detection-api -f"
echo "============================================================"
