# CLAUDE.md — PPE Detection Pipeline

## Overview

Dual-model YOLOv8 PPE detection pipeline: **YOLOv8s (COCO)** for person detection +
**yihong1120/Construction-Hazard-Detection (YOLO11m)** for PPE classification. Served
as a Flask web app. A person is **Safe** only when both a **Hardhat AND Safety Vest**
are matched with no negative override signals.

> **Validated:** 592 real 4K images (3840×2160). Result: 42 Safe, 72 Unsafe, 501 No-person, 0 Failed.

## Models

| Model | Source | Classes Used |
|-------|--------|-------------|
| Person | YOLOv8s COCO (`models/yolov8s.pt`, fallback `yolov8n.pt`) | `person` → `"Person"` |
| PPE | `yihong1120/Construction-Hazard-Detection` YOLO11m (HuggingFace) | `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest` |

The PPE model has 11 classes total — only the 4 above are used; Person/Mask/machinery/etc. are filtered out.

## Compliance Logic (`src/compliance.py`)

For each `Person` (confidence ≥ `PERSON_CONF_THRESHOLD`):
- **Safe (GREEN):** Hardhat matched AND Safety Vest matched, no negative signals.
- **Unsafe (RED):** Missing hardhat, missing vest, or `NO-Hardhat`/`NO-Safety Vest` override present.

### Matching Algorithm
- **Hardhats:** Center-containment in person's extended zone (15% upward margin for heads above body box). Nearest-neighbour greedy assignment prevents hat theft in overlapping zones.
- **Vests:** Center-containment in person's bounding box (no upward extension — vests sit on torso). Same nearest-neighbour greedy assignment.
- **Negative overrides:** `NO-Hardhat` center in extended zone → forces `has_helmet=False`. `NO-Safety Vest` center in person box → forces `has_vest=False`.

### Person Filters (applied before compliance)
| Filter | Config | Default | Purpose |
|--------|--------|---------|---------|
| Confidence | `PERSON_CONF_THRESHOLD` | `0.25` | Catch persons at distance |
| Min area | `MIN_PERSON_AREA_FRAC` | `0.001` | Reject pixel noise |
| Aspect ratio | `MIN_PERSON_ASPECT_RATIO` | `0.75` | Reject vehicles/equipment (wider than tall) |
| ROI | `SITE_ROI` | `""` (disabled) | Exclude persons outside site gate |

## Architecture

```
[Image] → YOLOv8s (COCO, person only) ──┐
         YOLO11m (PPE: hat/vest) ────────┤
                                         ├→ Compliance → Annotator → Output
```

### Flask Routes (`src/main.py`)
| Route | Method | Description |
|-------|--------|-------------|
| `/health` | GET | `{"status":"ok","model_loaded":true}` |
| `/detect` | POST | Multipart image → annotated JPEG (or JSON with `?json=1`) |
| `/` | GET | HTML upload form |

## Project Structure

```
config/settings.py          # All config from env vars
src/detector.py             # Dual-model PPEDetector singleton
src/compliance.py           # Hat+vest matching, compliance assessment
src/annotator.py            # Draws boxes + Safe/Unsafe labels
src/main.py                 # Flask app
src/watcher.py              # Phase 2 FTP folder watcher
src/utils/{image_io,bbox}.py
models/{yolov8s,yolov8n}.pt # Local person weights (gitignored)
scripts/run_batch_test.py   # Batch CLI: --input scripts/2025/ --output outputs/
tests/                      # 46 tests (pytest tests/ -v)
```

## Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `HF_MODEL_REPO` | `yihong1120/Construction-Hazard-Detection` | PPE model |
| `HF_MODEL_FILE` | `models/yolo11/pt/yolo11m.pt` | Weights path in repo |
| `PERSON_CONF_THRESHOLD` | `0.25` | |
| `PPE_CONF_THRESHOLD` | `0.35` | |
| `MIN_PERSON_AREA_FRAC` | `0.001` | |
| `MIN_PERSON_ASPECT_RATIO` | `0.75` | |
| `SITE_ROI` | `""` | `x1,y1,x2,y2` normalised fractions; empty = full frame |
| `DRAW_PPE_BOXES` | `true` | Yellow PPE boxes in Flask mode; batch uses `false` |
| `FLASK_PORT` | `5000` | |
| `UPLOAD_DIR` / `OUTPUT_DIR` / `DEAD_LETTER_DIR` | `uploads/` / `outputs/` / `failed/` | |
| `FTP_WATCH_DIR` / `POLL_INTERVAL_SEC` | `/ftp/uploads` / `2` | Phase 2 only |

## Annotation (`src/annotator.py`)

- **Safe:** Green box + `"Safe"` badge. **Unsafe:** Red box + `"Unsafe - PPE Hazard"` badge.
- Badge above box when space permits; inside box at top edge when near image top.
- Font scale proportional to box height for 4K readability.
- `DRAW_PPE_BOXES=true` draws individual PPE boxes in yellow.

## Commands

```bash
pip install -r requirements.txt
pytest tests/ -v --cov=src                                            # 46 tests
python scripts/run_batch_test.py --input scripts/2025/ --output outputs/  # Batch
flask --app src.main run --port 5000                                  # Dev
gunicorn -w 1 -b 0.0.0.0:5000 src.main:app                          # Prod
```

## Deployment (Digital Ocean)

1. Provision Ubuntu 22.04 (2 vCPU / 4 GB RAM). Clone repo, `pip install -r requirements.txt`.
2. Copy `models/yolov8s.pt` to server. Configure `.env` from `.env.example`. Set `SITE_ROI`.
3. First run auto-downloads PPE model from HuggingFace (~52 MB, cached).
4. `gunicorn -w 1 -b 0.0.0.0:5000 src.main:app` (single worker for model singleton).
5. Nginx reverse proxy on 80/443 for TLS. Run as non-root user.
6. (Phase 2) `FTP_WATCH_DIR` + `watcher.py` as systemd service.

## Key Design Decisions

- **Dual-model** because no single HF model has good Person + PPE detection together.
- **Center-containment** over IoU: hat box is ~5-15% of person box in 4K → IoU always < 0.2.
- **Nearest-neighbour greedy** prevents hat/vest theft in overlapping person zones.
- **Extended head zone** (15% upward) catches hats above person detection boundary.
- **SITE_ROI** excludes pedestrians/civilians outside the site gate.
- **NO-Hardhat / NO-Safety Vest overrides** — explicit absence > presence.

## Git Hygiene

`.env`, `*.pt`, `logs/`, `uploads/`, `outputs/`, `failed/`, `2025/`, `__pycache__/` gitignored.
Model weights never committed. Conventional Commits (`feat:`, `fix:`, `chore:`).

## Known Limitations

- PPE model also detects Mask/NO-Mask — currently filtered out (not used for compliance).
- Person detection from PPE model ignored in favour of proven YOLOv8s COCO results.
- For higher accuracy, custom fine-tuning on site-specific images (≥200) recommended.
