#!/usr/bin/env bash
# setup-ftp-watcher.sh — Install the PPE watcher on the FTP server.
#
# Usage (run as root on the FTP server 146.190.109.74):
#   bash setup-ftp-watcher.sh <API_DROPLET_IP>
#
# Example:
#   bash setup-ftp-watcher.sh 10.130.0.2

set -euo pipefail

API_IP="${1:?Usage: $0 <API_DROPLET_IP>}"
WATCHER_DIR="/opt/ppe-watcher"

echo "==> Installing Python requests..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip
pip3 install requests -q

echo "==> Setting up ${WATCHER_DIR}..."
mkdir -p "${WATCHER_DIR}"

# Copy watcher script
if [ -f "./ftp_watcher.py" ]; then
    cp ./ftp_watcher.py "${WATCHER_DIR}/ftp_watcher.py"
elif [ -f "./scripts/ftp_watcher.py" ]; then
    cp ./scripts/ftp_watcher.py "${WATCHER_DIR}/ftp_watcher.py"
else
    echo "ERROR: ftp_watcher.py not found. Copy it manually to ${WATCHER_DIR}/"
fi

echo "==> Installing systemd service..."
# Write service file with the actual API IP baked in
cat > /etc/systemd/system/ppe-ftp-watcher.service <<EOF
[Unit]
Description=PPE Detection FTP Watcher
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${WATCHER_DIR}
ExecStart=/usr/bin/python3 ftp_watcher.py --watch-dir /IHS-LAG-1197A --api-url http://${API_IP}:5000/detect/full --poll-interval 5
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo ""
echo "==> Testing connectivity to API..."
if curl -sf --connect-timeout 5 "http://${API_IP}:5000/health" > /dev/null 2>&1; then
    echo "    API is reachable at http://${API_IP}:5000/health"
else
    echo "    WARNING: Cannot reach API at http://${API_IP}:5000 — ensure the API droplet is running"
fi

echo ""
echo "============================================================"
echo "  Watcher installed. To start:"
echo ""
echo "  1. Test single scan first:"
echo "     python3 ${WATCHER_DIR}/ftp_watcher.py \\"
echo "       --watch-dir /IHS-LAG-1197A \\"
echo "       --api-url http://${API_IP}:5000/detect/full \\"
echo "       --once"
echo ""
echo "  2. Enable continuous polling:"
echo "     systemctl enable --now ppe-ftp-watcher"
echo ""
echo "  3. Monitor logs:"
echo "     journalctl -u ppe-ftp-watcher -f"
echo "============================================================"
