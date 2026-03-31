# Deployment Strategy — PPE Detection Pipeline

## 1. Microservice Assessment

The PPE Detection pipeline **already qualifies as a microservice**:

| Criterion | Status |
|-----------|--------|
| Single responsibility | PPE compliance detection only |
| Stateless request/response | Image in → result out, no session state |
| Own data model | No shared database |
| Independently deployable | Standalone Flask app with `/detect` and `/health` |
| Independent scaling | Can add instances behind a load balancer |

No architectural changes are needed to treat this as a microservice — just deploy it as one.

---

## 2. Deployment Architecture Options

### Option A: Standalone API Service (Simplest)

```
[Client / Agent / Other Module]
        │
        ▼ POST /detect
   ┌──────────┐
   │  PPE API  │  (Gunicorn + Nginx on a Droplet)
   └──────────┘
```

| | |
|---|---|
| **Pros** | Fast to deploy, minimal ops, low cost (~$24/mo for 4 GB droplet) |
| **Cons** | Single point of failure, manual scaling, no orchestration |
| **Best for** | PoC, low traffic, single-site deployment |

---

### Option B: API Behind a Gateway (Agent-Friendly) — Recommended

```
[Agent / Orchestrator / Other Microservices]
        │
        ▼
   ┌───────────────┐
   │  API Gateway   │  (Kong, Traefik, or Nginx)
   │  • Auth (keys) │
   │  • Rate limit  │
   │  • Routing     │
   │  • Versioning  │
   └───────┬───────┘
           ▼
   ┌──────────┐
   │  PPE API  │
   └──────────┘
```

| | |
|---|---|
| **Pros** | Auth, rate limiting, versioning, logging in one place. Other modules call the gateway, not the service directly. Easy to add more microservices later. |
| **Cons** | Slightly more setup than Option A |
| **Best for** | Multi-module system where an agent/orchestrator calls PPE as one of several tools |

**API contract exposed through gateway:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect` | POST | Multipart image → annotated JPEG (or JSON with `?json=1`) |
| `/api/v1/health` | GET | `{"status":"ok","model_loaded":true}` |

Authentication via `X-API-Key` header, rate limited per key.

---

### Option C: Kubernetes (Full Orchestration)

```
[Ingress Controller]
        │
        ▼
   ┌─────────────────────────┐
   │  K8s Cluster (DOKS)      │
   │  ┌─────────┐ ┌─────────┐│
   │  │ PPE Pod │ │ PPE Pod ││  (HPA scales on CPU)
   │  └─────────┘ └─────────┘│
   │  ┌─────────────────────┐│
   │  │  Other services     ││
   │  └─────────────────────┘│
   └─────────────────────────┘
```

| | |
|---|---|
| **Pros** | Auto-scaling, self-healing, rolling deploys, service mesh, multi-service orchestration |
| **Cons** | Significant complexity, DigitalOcean K8s starts at ~$48/mo + node costs, overkill for 1-2 services |
| **Best for** | Multiple microservices, high availability requirements, 2000+ sites |

---

## 3. Decision Matrix

| Question | If Yes → | If No → |
|----------|----------|---------|
| Will other modules/agents call this service? | Stable API contract + auth (Option B+) | Option A is fine |
| Do you need auto-scaling? | K8s or DO App Platform | Single droplet works |
| Are there other microservices planned? | K8s starts to make sense | Don't introduce K8s for one service |
| Is this a PoC or production? | Production → at least Option B | PoC → Option A, ship today |
| What's the expected request volume? | High/bursty → K8s | Low/steady → single droplet |

**Recommendation:** Start with **Option B**. Deploy on a droplet with Nginx acting as the API gateway (auth + rate limiting). This gives other modules and agents a stable integration point. If you later need K8s, the service is already containerized-ready (add a Dockerfile) and the API contract doesn't change.

---

## 4. Step-by-Step Deployment (DigitalOcean Droplet)

### Step 1 — Create the Droplet

- **Image:** Ubuntu 22.04 LTS
- **Size:** 2 vCPU / 4 GB RAM minimum (see [R&D.md](R&D.md) for sizing at scale)
- **Region:** Closest to your users
- Add your SSH key during creation

### Step 2 — Initial Server Setup

```bash
ssh root@<DROPLET_IP>
adduser ppe && usermod -aG sudo ppe
ufw allow OpenSSH && ufw allow 80 && ufw allow 443 && ufw enable
```

### Step 3 — Install Dependencies

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git nginx libgl1 libglib2.0-0
```

> `libgl1` and `libglib2.0-0` are required by OpenCV.

### Step 4 — Clone & Setup App

```bash
su - ppe
git clone <YOUR_REPO_URL> ~/ppe-detection
cd ~/ppe-detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install gunicorn
```

### Step 5 — Copy Model Weights

From your local machine:

```bash
scp models/yolov8s.pt ppe@<DROPLET_IP>:~/ppe-detection/models/
```

The PPE model (`yolo11m.pt`) auto-downloads from HuggingFace on first run (~52 MB, cached).

### Step 6 — Configure Environment

```bash
cp .env.example .env
nano .env
```

Set at minimum:

| Variable | Value |
|----------|-------|
| `SITE_ROI` | `0.0,0.10,0.85,1.0` (adjust for your site) |
| `DRAW_PPE_BOXES` | `true` |
| `FLASK_PORT` | `5000` |

See [CLAUDE.md](CLAUDE.md) for the full environment variable reference.

### Step 7 — Smoke Test

```bash
source venv/bin/activate
gunicorn -w 1 -b 0.0.0.0:5000 src.main:app
```

Hit `http://<DROPLET_IP>:5000/health` from your browser — expect `{"status":"ok","model_loaded":true}`.

### Step 8 — Systemd Service

```bash
sudo nano /etc/systemd/system/ppe.service
```

```ini
[Unit]
Description=PPE Detection API
After=network.target

[Service]
User=ppe
WorkingDirectory=/home/ppe/ppe-detection
Environment="PATH=/home/ppe/ppe-detection/venv/bin"
EnvironmentFile=/home/ppe/ppe-detection/.env
ExecStart=/home/ppe/ppe-detection/venv/bin/gunicorn -w 1 -b 127.0.0.1:5000 src.main:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ppe
```

### Step 9 — Nginx Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/ppe
```

```nginx
server {
    listen 80;
    server_name <DROPLET_IP_OR_DOMAIN>;
    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/ppe /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx
```

### Step 10 — TLS with Let's Encrypt (Optional)

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## 5. Upgrade Path

```
Option A (Standalone)          you are here ──► deploy today
       │
       ▼ add auth + rate limiting
Option B (API Gateway)         ──► stable integration point for agents
       │
       ▼ add Dockerfile + K8s manifests
Option C (Kubernetes)          ──► auto-scaling, multi-service
```

Each step is additive — no rewrites required. The `/detect` API contract stays the same throughout.

---

## 6. Further Reading

- **Scaling & optimization:** See [R&D.md](R&D.md) for ONNX Runtime, INT8 quantization, 500-site architecture, and cost estimates.
- **Configuration reference:** See [CLAUDE.md](CLAUDE.md) for all environment variables, model details, and compliance logic.
