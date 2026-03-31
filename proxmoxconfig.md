# Proxmox VM Configuration — PPE Detection Pipeline

## 1. Identify Your AMD Opteron Model

Before creating the VM, determine your exact CPU. From the **Proxmox web UI**:

**Datacenter → Your Node → Summary** — CPU model shown at top.

Or from **Proxmox shell** (Node → Shell):

```bash
# CPU model
cat /proc/cpuinfo | grep "model name" | head -1

# Instruction sets (critical for PyTorch compatibility)
grep -o -E 'avx2?|sse4_[12]|fma' /proc/cpuinfo | sort -u

# Core count
nproc
```

### Compatibility Matrix

| Series | Example Model | AVX | SSE4.1/4.2 | PyTorch (pip) | ONNX Runtime | Recommended Path |
|--------|--------------|-----|-------------|---------------|-------------|-----------------|
| **6100** | 6128, 6172, 6176 | No | No | **CRASHES** (Illegal instruction) | Works | ONNX Runtime only |
| **6200** | 6212, 6238, 6278 | Yes (slow) | Yes | Works | Works | PyTorch OK, ONNX faster |
| **6300** | 6320, 6344, 6380 | Yes (slow) | Yes | Works | Works | PyTorch OK, ONNX faster |

**If `avx` appears in flags** → PyTorch works. **If it doesn't** → you MUST use the ONNX Runtime path (see Section 6).

---

## 2. Proxmox VM Creation

### Hardware Settings

Create the VM via **Datacenter → Create VM** with these settings:

| Setting | Minimum | Recommended | Notes |
|---------|---------|-------------|-------|
| **CPU Type** | `host` | `host` | **Critical.** Passes through all CPU flags (AVX/SSE4). Never use `kvm64` (strips AVX). |
| **CPU Cores** | 4 | 8 | Opteron modules share FPU — assign 2x the module count you want. |
| **CPU Sockets** | 1 | 1 | Match physical topology if NUMA enabled. |
| **RAM** | 4 GB | 8 GB | Models consume ~400 MB; 4K image processing peaks at ~600 MB transient. |
| **Disk** | 10 GB | 20 GB | OS (~3 GB) + venv (~400 MB) + models (~180 MB) + logs + output staging. |
| **Network** | virtio | virtio | Bridge to LAN. First-run HuggingFace download needs internet (~52 MB). |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | Or Debian 12. Avoid CentOS (older Python). |
| **BIOS** | SeaBIOS | SeaBIOS | OVMF (UEFI) works but unnecessary overhead. |
| **Machine** | q35 | q35 | Modern chipset emulation. |

### Proxmox VM Config File

The VM config lives at `/etc/pve/qemu-server/<VMID>.conf`. Key lines:

```ini
# Example for VMID 110
cpu: host
cores: 8
sockets: 1
memory: 8192
balloon: 4096
numa: 1
net0: virtio,bridge=vmbr0
scsi0: local-lvm:vm-110-disk-0,size=20G
ostype: l26
```

### Setting CPU Type to `host` (if not set during creation)

```bash
# From Proxmox shell
qm set 110 -cpu host
```

Or in **Web UI**: VM → Hardware → Processors → Type → select **host**.

---

## 3. Advanced Proxmox Settings

### NUMA (Non-Uniform Memory Access)

If your Opteron server has multiple sockets (common in 2U rack servers):

```bash
qm set 110 -numa 1
```

This tells QEMU to expose NUMA topology to the guest, improving memory locality. Particularly important on Opteron G34 dual-socket boards where cross-socket memory access is 30-50% slower.

### Memory Ballooning

```bash
qm set 110 -balloon 4096 -memory 8192
```

- `memory`: Maximum RAM the VM can use (8 GB).
- `balloon`: Minimum guaranteed RAM (4 GB). Proxmox can reclaim the rest for other VMs when idle.
- During inference peaks, the VM will use up to 8 GB. During idle, it releases back to 4 GB.

### CPU Pinning (Optional, Advanced)

If the host runs other workloads and you want to isolate PPE inference:

```bash
# Pin VM 110 to physical cores 0-7
qm set 110 -cpulimit 8
taskset -acp 0-7 $(cat /run/qemu-server/110.pid)
```

This prevents inference from competing with other VMs for cache.

---

## 4. Guest OS Setup (Inside the VM)

```bash
# Update and install Python
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv git

# Create app directory
mkdir -p /opt/ppe
cd /opt/ppe

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Transfer project files (via FileZilla/scp) then install
pip install -r requirements.txt

# Create runtime directories
mkdir -p uploads outputs failed logs

# Copy .env
cp .env.example .env
nano .env  # Set FLASK_ENV=production, SITE_ROI, paths
```

### Verify CPU Compatibility Inside VM

```bash
# Check that host CPU flags are visible
grep -o -E 'avx2?|sse4_[12]' /proc/cpuinfo | sort -u

# Test PyTorch loads without crashing
python3 -c "import torch; print(f'PyTorch OK, threads={torch.get_num_threads()}')"

# Test model loading
cd /opt/ppe
source venv/bin/activate
python3 -c "from src.detector import PPEDetector; PPEDetector.get_instance(); print('Models loaded OK')"
```

**If PyTorch crashes with "Illegal instruction"** → your CPU lacks AVX. Follow Section 6 (ONNX Runtime path).

---

## 5. CPU Optimization Tuning

### OpenMP Thread Count

Opteron Bulldozer/Piledriver uses CMT (Clustered Multi-Threading) where each module has 2 integer cores but 1 shared FPU. For FP-heavy inference, **set threads = number of modules, not cores**:

```bash
# Add to /opt/ppe/.env or systemd service
OMP_NUM_THREADS=4          # For 8-core VM (4 modules)
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
```

Or in Python at startup:
```python
import torch
torch.set_num_threads(4)   # Match to module count
```

**Rule of thumb:** `OMP_NUM_THREADS = assigned_cores / 2` on Opteron 6200/6300.

### Memory Allocation

```bash
# Add to /opt/ppe/.env
MALLOC_ARENA_MAX=2         # Prevent glibc arena fragmentation
```

### Disable Hyper-Threading Awareness (if applicable)

```bash
# In systemd service Environment
GOMP_CPU_AFFINITY="0-3"    # Bind to first 4 physical cores only
```

---

## 6. ONNX Runtime Path (Required for Opteron 6100, Optional for 6200/6300)

If PyTorch crashes or you want 1.5-3x faster inference, export models to ONNX and use ONNX Runtime.

### Step 1: Export Models (on your development machine, not the Opteron)

```bash
cd "C:\Users\Admin\Documents\PPE Detection"
python -c "
from ultralytics import YOLO

# Export person model
person = YOLO('models/yolov8s.pt')
person.export(format='onnx', imgsz=640, simplify=True, opset=17)
print('Exported: models/yolov8s.onnx')

# Export PPE model
from huggingface_hub import hf_hub_download
weights = hf_hub_download('yihong1120/Construction-Hazard-Detection', 'models/yolo11/pt/yolo11m.pt')
ppe = YOLO(weights)
ppe.export(format='onnx', imgsz=640, simplify=True, opset=17)
print('Exported: yolo11m.onnx')
"
```

### Step 2: Transfer ONNX Files to Server

Copy both `.onnx` files to `models/` on the server via FileZilla/scp.

### Step 3: Install ONNX Runtime on Server

```bash
# Replace ultralytics PyTorch inference with ONNX Runtime
pip install onnxruntime
```

### Step 4: Update detector.py to Use ONNX

Ultralytics natively supports ONNX inference — just point it at the `.onnx` file:

```python
# In src/detector.py, _load_person_model():
model = YOLO("models/yolov8s.onnx")  # Instead of .pt

# In _load_ppe_model():
model = YOLO("models/yolo11m.onnx")  # Instead of downloading .pt
```

No other code changes needed — ultralytics detects ONNX format and uses `onnxruntime` automatically.

### Expected Speedup

| Opteron Series | PyTorch (ms/image) | ONNX Runtime (ms/image) | Speedup |
|---------------|-------------------|------------------------|---------|
| 6100 (no AVX) | Crashes | 800-1500 | N/A |
| 6200 (AVX) | 1000-1800 | 400-800 | ~2x |
| 6300 (AVX) | 800-1500 | 300-700 | ~2x |

---

## 7. Systemd Service (Inside VM)

```bash
nano /etc/systemd/system/ppe.service
```

```ini
[Unit]
Description=PPE Detection Flask App
After=network.target

[Service]
User=root
WorkingDirectory=/opt/ppe
Environment="PATH=/opt/ppe/venv/bin"
Environment="OMP_NUM_THREADS=4"
Environment="MALLOC_ARENA_MAX=2"
ExecStart=/opt/ppe/venv/bin/gunicorn -w 1 -b 0.0.0.0:5000 --timeout 30 src.main:app
Restart=always
RestartSec=5
StandardOutput=append:/opt/ppe/logs/app.log
StandardError=append:/opt/ppe/logs/app.log

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable ppe
systemctl start ppe
```

---

## 8. Health Monitoring

### Basic Checks

```bash
# Is the service running?
systemctl status ppe

# Is inference working?
curl -s http://localhost:5000/health | python3 -m json.tool

# Live logs
tail -f /opt/ppe/logs/app.log
```

### Resource Monitoring

```bash
# CPU and memory usage of the PPE process
ps aux | grep gunicorn

# Real-time resource monitor
htop -p $(pgrep -f gunicorn | tr '\n' ',')

# Disk usage
du -sh /opt/ppe/outputs/ /opt/ppe/logs/
```

### Log Rotation

```bash
nano /etc/logrotate.d/ppe
```

```
/opt/ppe/logs/*.log {
    daily
    rotate 14
    compress
    missingok
    notifempty
    postrotate
        systemctl restart ppe
    endscript
}
```

---

## 9. Quick Reference — VM Specs Summary

| Resource | Minimum (tight) | Recommended (comfortable) |
|----------|-----------------|--------------------------|
| CPU Cores | 4 | 8 |
| CPU Type | `host` | `host` |
| RAM | 4 GB | 8 GB |
| Disk | 10 GB | 20 GB |
| Network | 10 Mbps | 100 Mbps |
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| Python | 3.10+ | 3.12 |
| Inference | ~1-2 sec/image (ONNX) | ~0.5-1 sec/image (ONNX + tuned) |
