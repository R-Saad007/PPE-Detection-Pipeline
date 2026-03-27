# PPE Detection Pipeline

Personal Protective Equipment (PPE) compliance detection for construction sites using a dual-model YOLOv8 architecture. The system processes camera images, detects persons, evaluates whether each person is wearing a **hardhat** and **safety vest**, and returns annotated images labelled **"Safe"** or **"Unsafe - PPE Hazard"**.

Designed for deployment on a Digital Ocean droplet receiving images from site cameras via FTP.

---

## Table of Contents

- [Architecture](#architecture)
- [Models](#models)
- [Compliance Logic](#compliance-logic)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Batch Processing](#batch-processing)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Integration Notes](#integration-notes)
- [Known Limitations](#known-limitations)

---

## Architecture

Two YOLO models run in sequence per image:

```
                    ┌──────────────────────────┐
                    │       Input Image         │
                    └────────────┬─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                                      ▼
   ┌─────────────────────┐              ┌──────────────────────────┐
   │  YOLOv8s (COCO)     │              │  YOLO11m (yihong1120)    │
   │  Person detection    │              │  PPE classification      │
   │  Class: person → 0   │              │  Hardhat, NO-Hardhat,    │
   │                      │              │  Safety Vest,             │
   │                      │              │  NO-Safety Vest           │
   └──────────┬──────────┘              └────────────┬─────────────┘
              │                                      │
              └──────────────────┬───────────────────┘
                                 ▼
                  ┌──────────────────────────┐
                  │   Compliance Engine      │
                  │   Hat + Vest matching    │
                  │   per detected person    │
                  └────────────┬─────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │   Annotator              │
                  │   Green = Safe           │
                  │   Red = Unsafe           │
                  └──────────────────────────┘
```

**Why two models?** No single HuggingFace model reliably detects both persons and PPE. The COCO-trained YOLOv8s provides robust person detection across varying distances and angles in 4K images, while the `yihong1120/Construction-Hazard-Detection` YOLO11m model specialises in PPE classification with explicit absence signals (`NO-Hardhat`, `NO-Safety Vest`).

---

## Models

### Person Detection — YOLOv8s (COCO)

- **Weights:** `models/yolov8s.pt` (~22 MB, local). Falls back to `models/yolov8n.pt` (~6 MB) if unavailable.
- **Class used:** COCO class 0 (`person`), normalised to label `"Person"`.
- **Why YOLOv8s over YOLOv8n:** Better recall on persons at varying distances and angles in 4K (3840x2160) images. Validated on 592 real camera images.

### PPE Detection — YOLO11m (yihong1120/Construction-Hazard-Detection)

- **Source:** [HuggingFace — yihong1120/Construction-Hazard-Detection](https://huggingface.co/yihong1120/Construction-Hazard-Detection)
- **Weights:** `models/yolo11/pt/yolo11m.pt` (auto-downloaded to HuggingFace cache on first run, ~52 MB).
- **Full class map (11 classes):**

  | ID | Label | Used for Compliance |
  |----|-------|-------------------|
  | 0 | Hardhat | Required PPE |
  | 1 | Mask | Filtered out |
  | 2 | NO-Hardhat | Absence override |
  | 3 | NO-Mask | Filtered out |
  | 4 | NO-Safety Vest | Absence override |
  | 5 | Person | Filtered out (COCO model used instead) |
  | 6 | Safety Cone | Filtered out |
  | 7 | Safety Vest | Required PPE |
  | 8 | machinery | Filtered out |
  | 9 | utility pole | Filtered out |
  | 10 | vehicle | Filtered out |

Only classes 0, 2, 4, 7 (`Hardhat`, `NO-Hardhat`, `NO-Safety Vest`, `Safety Vest`) are passed to the compliance engine. All others are discarded at the detection stage.

---

## Compliance Logic

For each detected `Person` bounding box that passes all filters:

| Condition | Result | Colour | Annotation |
|-----------|--------|--------|------------|
| Hardhat matched AND Safety Vest matched, no negative signals | **Safe** | Green | `Safe` |
| Missing hardhat, missing vest, or negative override present | **Unsafe - PPE Hazard** | Red | `Unsafe - PPE Hazard` + missing items listed below |

### PPE-to-Person Matching

**Hardhats** are matched using **center-containment**: the hat's center point must fall within the person's bounding box extended 15% upward (to account for heads above the body detection boundary). When multiple person zones overlap, **nearest-neighbour greedy assignment** ensures each hat goes to the geometrically closest person — preventing "hat theft" in crowd scenes.

**Safety Vests** use the same algorithm but match against the person's original bounding box (no upward extension — vests sit on the torso, not the head).

### Negative Override Signals

If a `NO-Hardhat` detection's center falls within a person's extended zone, that person's `has_helmet` is forced to `False` — even if a `Hardhat` was also matched. Same logic applies for `NO-Safety Vest` within the person box.

### Person Filters

Four filters are applied to raw person detections before compliance evaluation:

| Filter | Environment Variable | Default | Purpose |
|--------|---------------------|---------|---------|
| Confidence | `PERSON_CONF_THRESHOLD` | `0.25` | Catch persons at distance/angle |
| Minimum area | `MIN_PERSON_AREA_FRAC` | `0.001` | Reject pixel noise / tiny artefacts |
| Aspect ratio | `MIN_PERSON_ASPECT_RATIO` | `0.75` | Reject vehicles/equipment (wider than tall) |
| ROI containment | `SITE_ROI` | `""` (disabled) | Exclude persons outside site boundary |

---

## Getting Started

### Prerequisites

- Python 3.10+
- ~4 GB disk space (models + dependencies)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ppe-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed (defaults work for local development)

# Copy person model weights (not in repo)
# Place yolov8s.pt (and optionally yolov8n.pt) in models/
```

### First Run

```bash
# Start the Flask app (downloads PPE model from HuggingFace on first run)
flask --app src.main run --port 5000

# Verify it's running
curl http://localhost:5000/health
# → {"model_loaded": true, "status": "ok"}

# Test with an image
curl -X POST http://localhost:5000/detect \
     -F "image=@path/to/test_image.jpg" \
     --output annotated_result.jpg
```

---

## Configuration

All settings are controlled via environment variables (loaded from `.env` via `python-dotenv`). See `.env.example` for the complete list.

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | `development` or `production` |
| `FLASK_PORT` | `5000` | HTTP port |
| `HF_MODEL_REPO` | `yihong1120/Construction-Hazard-Detection` | HuggingFace PPE model repository |
| `HF_MODEL_FILE` | `models/yolo11/pt/yolo11m.pt` | Weights file path within the repo |
| `PERSON_CONF_THRESHOLD` | `0.25` | Min confidence for person detections |
| `PPE_CONF_THRESHOLD` | `0.35` | Min confidence for PPE detections |
| `MIN_PERSON_AREA_FRAC` | `0.001` | Min person box area as fraction of image |
| `MIN_PERSON_ASPECT_RATIO` | `0.75` | Min height/width ratio (rejects vehicles) |
| `SITE_ROI` | `""` (full frame) | Site boundary: `x1,y1,x2,y2` normalised fractions |
| `DRAW_PPE_BOXES` | `true` | Draw individual PPE boxes in yellow (debug) |
| `UPLOAD_DIR` | `uploads/` | Temporary upload storage |
| `OUTPUT_DIR` | `outputs/` | Annotated output directory |
| `DEAD_LETTER_DIR` | `failed/` | Quarantine for unprocessable images |
| `MAX_IMAGE_SIZE_MB` | `20` | Reject uploads above this size |
| `FTP_WATCH_DIR` | `/ftp/uploads` | (Phase 2) FTP directory to poll |
| `POLL_INTERVAL_SEC` | `2` | (Phase 2) Watcher polling interval |

### SITE_ROI Configuration

`SITE_ROI` defines a rectangular region (normalised fractions of image dimensions). Persons whose bounding-box center falls outside this region are excluded as pedestrians/civilians and are **not detected at all** — no bounding box, no label.

**Current production value** (validated on IHS-LAG-1197A camera):

```bash
# Excludes persons beyond the fence/gate (right 15%) and rooftop area (top 10%)
SITE_ROI=0.0,0.10,0.85,1.0
```

```bash
# Full frame (disabled — all persons checked)
SITE_ROI=
```

**How to determine values for a new camera:**
1. Open a sample image and identify the site boundary (fence, gate, wall)
2. Measure in pixels, then divide by image dimensions
3. For 3840x2160: `x_frac = x_px / 3840`, `y_frac = y_px / 2160`
4. Format: `SITE_ROI=x1,y1,x2,y2` where `(x1,y1)` is the top-left and `(x2,y2)` is the bottom-right of the monitored zone

---

## API Reference

### `GET /health`

Liveness probe for uptime monitoring.

**Response:**
```json
{"status": "ok", "model_loaded": true}
```

### `POST /detect`

Accepts a multipart/form-data image upload. Returns an annotated JPEG with compliance labels.

**Request:**
```bash
curl -X POST http://localhost:5000/detect \
     -F "image=@photo.jpg" \
     --output result.jpg
```

**Query Parameters:**
- `?json=1` — Return JSON metadata instead of annotated image.

**JSON Response (when `?json=1`):**
```json
{
  "status": "Unsafe - PPE Hazard",
  "persons_detected": 2,
  "safe": 1,
  "unsafe": 1
}
```

**Status values:** `"Safe"`, `"Unsafe - PPE Hazard"`, `"No person detected"`

**Error responses:** `400` (bad input), `500` (inference error)

### `GET /`

Minimal HTML upload form for manual browser testing.

---

## Annotation Output

### Label Format

**Safe persons** receive a single-line green badge:

```
┌──────────────┐
│ Safe         │
└──────────────┘
```

**Unsafe persons** receive a multi-line red badge listing the specific missing equipment:

```
┌───────────────────────┐
│ Unsafe - PPE Hazard   │
│   No Safety Helmet    │
│   No Safety Vest      │
└───────────────────────┘
```

If only one item is missing, only that line appears. Both lines appear when both are absent.

### Visual Style

- **Safe:** Green bounding box + green badge with white `"Safe"` text.
- **Unsafe:** Red bounding box + red badge with white text. Missing equipment listed as indented sub-lines (85% font scale, same stroke weight as the main label for legibility).
- **Badge position:** Above the bounding box when space permits. Falls back to inside the box at its top edge when the person is near the top of the image.
- **Font:** `cv2.FONT_HERSHEY_SIMPLEX`. Scale proportional to box height for readability at any resolution (4K → 1080p).
- **PPE overlay:** When `DRAW_PPE_BOXES=true`, individual PPE bounding boxes are drawn in yellow. Default `true` in Flask mode; batch mode uses `false` for clean client-review output.

---

## Batch Processing

Process a directory of images without running the Flask server:

```bash
# Run inference on all images, write annotated outputs
python scripts/run_batch_test.py --input scripts/2025/ --output outputs/

# Dry run — list images without processing
python scripts/run_batch_test.py --input scripts/2025/ --dry-run
```

- Recursively walks the input directory for `.jpg` / `.JPG` files.
- Skips non-image files (`.txt` camera metadata) silently.
- Validates JPEG magic bytes before inference.
- Mirrors the source directory structure in the output.
- Appends `_annotated` to output filenames.
- Moves failed images to `DEAD_LETTER_DIR`.
- Draws person box + compliance label only (no individual PPE boxes).

### Validated Results (592 real 4K images, SITE_ROI active)

| Status | Count |
|--------|-------|
| Safe | 42 |
| Unsafe - PPE Hazard | 71 |
| No person detected | 501 |
| Failed | 0 |

---

## Testing

```bash
# Run all tests (49 tests)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_compliance.py -v
```

### Test Coverage

| Module | Tests |
|--------|-------|
| `test_compliance.py` (22) | Safe (hat+vest), Unsafe (missing either), vest-alone, hat-alone, negative overrides, NO-Safety Vest override, multi-person, no-double-counting (hats and vests), nearest-neighbour in overlapping zones |
| `test_annotator.py` (11) | Green/red colours, badge placement, box clipping, PPE box toggle, multi-line unsafe badge (missing helmet, missing vest, missing both) |
| `test_detector.py` (8) | Detection NamedTuple, label constants, singleton, dual-model merge |
| `test_integration.py` (8) | Full pipeline (safe, unsafe, no-person), Flask routes (`/health`, `/detect`, `/`), JSON response mode |

---

## Project Structure

```
ppe-detection/
├── CLAUDE.md                    # AI assistant context (condensed architecture reference)
├── README.md                    # This file
├── .env.example                 # Environment variable template
├── .gitignore
├── requirements.txt             # Pinned dependencies
│
├── config/
│   ├── settings.py              # Centralised config — all values from env vars
│   └── logging_config.py        # Structured JSON logging setup
│
├── models/
│   ├── classes.yaml             # Label → compliance role mapping (reference)
│   ├── yolov8s.pt               # Person detection weights (~22 MB, gitignored)
│   └── yolov8n.pt               # Fallback person weights (~6 MB, gitignored)
│   # PPE model (yolo11m.pt) auto-downloaded to ~/.cache/huggingface/hub/
│
├── src/
│   ├── __init__.py
│   ├── main.py                  # Flask app — routes: /health, /detect, /
│   ├── detector.py              # Dual-model PPEDetector singleton
│   ├── compliance.py            # Hat+vest matching, compliance assessment
│   ├── annotator.py             # Draws boxes and Safe/Unsafe labels
│   ├── watcher.py               # (Phase 2) FTP directory poller
│   └── utils/
│       ├── __init__.py
│       ├── image_io.py          # Image loading, saving, validation
│       └── bbox.py              # IoU, center-containment helpers
│
├── tests/
│   ├── conftest.py              # Shared fixtures (person, hat, vest, image)
│   ├── test_compliance.py       # 22 compliance logic tests
│   ├── test_annotator.py        # 11 annotation tests
│   ├── test_detector.py         # 8 detector tests
│   ├── test_integration.py      # 8 integration + Flask route tests
│   └── fixtures/                # Synthetic test images
│
├── scripts/
│   ├── run_batch_test.py        # Batch inference CLI
│   ├── benchmark.py             # Throughput profiling
│   └── 2025/                    # Real camera images (gitignored)
│       ├── 10/                  # Camera group 10
│       └── 11/                  # Camera group 11
│
├── uploads/                     # Temporary upload storage (gitignored)
├── outputs/                     # Annotated results (gitignored)
├── failed/                      # Dead-letter quarantine (gitignored)
└── logs/                        # Runtime logs (gitignored)
```

---

## Deployment

### Requirements

- Ubuntu 22.04 LTS (or similar)
- 2 vCPU / 4 GB RAM (minimum for YOLOv8s + YOLO11m in memory)
- Python 3.10+
- ~500 MB disk for models + HuggingFace cache

### Step-by-Step

```bash
# 1. Clone and install
git clone <repo-url> /opt/ppe
cd /opt/ppe
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy person model weights
cp /path/to/yolov8s.pt models/

# 3. Configure
cp .env.example .env
nano .env  # Set FLASK_ENV=production, SITE_ROI, paths

# 4. Create runtime directories
mkdir -p uploads outputs failed logs

# 5. Test (downloads PPE model on first run)
python -c "from src.detector import PPEDetector; PPEDetector.get_instance(); print('OK')"

# 6. Run with gunicorn (single worker — models are singletons)
gunicorn -w 1 -b 0.0.0.0:5000 src.main:app
```

### Production Setup

**Systemd service** (`/etc/systemd/system/ppe.service`):
```ini
[Unit]
Description=PPE Detection Flask App
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/ppe
Environment="PATH=/opt/ppe/venv/bin"
ExecStart=/opt/ppe/venv/bin/gunicorn -w 1 -b 0.0.0.0:5000 src.main:app
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

**Nginx reverse proxy** (port 80/443 → 5000):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 25M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 30s;
    }
}
```

### Phase 2 — FTP Watcher

Once FTP is provisioned on the server, run `watcher.py` as a separate systemd service:

```bash
FTP_WATCH_DIR=/ftp/uploads POLL_INTERVAL_SEC=2 python -m src.watcher
```

The watcher polls the FTP directory for new images, processes them through the same detection pipeline, and writes annotated results to `OUTPUT_DIR`.

---

## Integration Notes

### For Platform Developers

**Programmatic usage (without Flask):**

```python
from src.detector import PPEDetector
from src.compliance import assess_compliance
from src.annotator import annotate
import cv2

# Load models once at startup
detector = PPEDetector.get_instance()

# Per-image processing
img = cv2.imread("photo.jpg")
detections = detector.detect(img)          # List[Detection]
results = assess_compliance(detections)    # List[ComplianceResult]
annotated = annotate(img, results)         # Annotated numpy array
cv2.imwrite("output.jpg", annotated)

# Access compliance data
for r in results:
    print(r.person.box)     # (x1, y1, x2, y2)
    print(r.is_safe)        # True/False
    print(r.has_helmet)     # True/False
    print(r.has_vest)       # True/False
    print(r.label)          # "Safe" or "Unsafe - PPE Hazard"
```

**Key data types:**

```python
# Detection (NamedTuple)
Detection(label: str, confidence: float, x1: float, y1: float, x2: float, y2: float)
# .box property returns (x1, y1, x2, y2) tuple

# ComplianceResult (dataclass)
ComplianceResult(person: Detection, is_safe: bool, has_vest: bool, has_helmet: bool)
# .label property returns "Safe" or "Unsafe - PPE Hazard"
```

**Model singleton:** `PPEDetector.get_instance()` loads both models into memory (~400 MB total). Call once at application startup — subsequent calls return the cached instance. Use a single gunicorn worker to keep both models in memory.

**Thread safety:** The detector is not thread-safe. Use a single-threaded worker (gunicorn `-w 1`) or add locking if integrating into a multi-threaded application.

**Image format:** All functions expect BGR `uint8` NumPy arrays (OpenCV default). If your pipeline uses RGB (Pillow/matplotlib), convert with `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` before passing to `detect()`.

**JSON API:** For services that need structured data rather than annotated images, use `POST /detect?json=1` which returns person counts and compliance status without image processing overhead.

---

## Known Limitations

- **No mask/goggles compliance.** The PPE model detects Mask/NO-Mask but these are currently filtered out. Can be enabled by adding to the label filter in `detector.py` and extending `ComplianceResult`.
- **Single-worker constraint.** Models consume ~400 MB RAM as singletons. Multi-worker deployments would load models per worker. Use a single gunicorn worker.
- **No GPU acceleration configured.** Runs on CPU by default. For GPU, install `torch` with CUDA support and ultralytics will use it automatically.
- **Person detection from PPE model ignored.** The YOLO11m model detects `Person` but its results are filtered out in favour of the proven YOLOv8s COCO model. This could be revisited if the PPE model's person detection proves comparable.
- **Custom fine-tuning.** For higher accuracy on site-specific images, fine-tune on >= 200 labelled images from the target cameras and replace the HuggingFace model weights.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `flask` | 3.1.3 | Web framework |
| `ultralytics` | 8.4.26 | YOLO model loading and inference |
| `huggingface_hub` | 0.27.1 | PPE model download |
| `opencv-python-headless` | 4.10.0.84 | Image processing |
| `numpy` | >= 1.26.4 | Array operations |
| `python-dotenv` | 1.0.1 | Environment variable loading |
| `gunicorn` | 23.0.0 | Production WSGI server |
| `pytest` | 8.3.3 | Testing framework |
| `pytest-cov` | 5.0.0 | Coverage reporting |
| `pip-audit` | 2.7.3 | Dependency vulnerability scanning |

---

## Security

- Credentials and paths are never hardcoded — all from environment variables via `.env` (gitignored).
- Uploaded file paths are sanitised with `Path.resolve()` and verified within `UPLOAD_DIR`.
- Image inputs validated by extension AND magic bytes before inference.
- Maximum upload size enforced (`MAX_IMAGE_SIZE_MB`).
- Failed/corrupt images quarantined in `DEAD_LETTER_DIR` (never deleted, never re-processed).
- Flask runs as a non-root user in production.
- Dependencies pinned in `requirements.txt`. Run `pip-audit` to check for vulnerabilities.

---

## License

Internal use only. Not for public distribution.
