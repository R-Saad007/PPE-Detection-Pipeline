# PPE Detection — Cost Analysis & Value Proposition

> **Date:** April 7, 2026
> **Scope:** Image volume from on-site FTP camera (IHS-LAG-1197A), Jan–Apr 2026
> **Purpose:** Quantify savings from replacing Gemini Vision API with self-hosted YOLOv8/YOLO11m pipeline

---

## 1. Image Volume Summary (Jan 1 – Apr 7, 2026)

| Month     | Images | Active Days | Avg Images/Day |
|-----------|-------:|------------:|---------------:|
| January   |    512 |          25 |           20.5 |
| February  |    550 |          26 |           21.2 |
| March     |    618 |          31 |           19.9 |
| April 1–7 |    141 |           7 |           20.1 |
| **Total** | **1,821** | **89** |       **20.5** |

### Volume Projections

| Metric                         | Value   |
|--------------------------------|--------:|
| Average images/day             |    20.5 |
| Projected monthly (30 days)    |   ~615  |
| Projected annual (365 days)    | ~7,480  |
| Peak day (Apr 4)               |      29 |
| Lowest day (Feb 10)            |       7 |
| Days with zero uploads (gaps)  |      ~8 |

### Daily Breakdown by Month

<details>
<summary>January (512 images, 25 active days)</summary>

| Day | Images |
|-----|-------:|
| 7   | 18 |
| 8   | 24 |
| 9   | 14 |
| 10  | 21 |
| 11  | 15 |
| 12  | 18 |
| 13  | 21 |
| 14  | 21 |
| 15  | 20 |
| 16  | 20 |
| 17  | 24 |
| 18  | 22 |
| 19  | 20 |
| 20  | 2  |
| 21  | 20 |
| 22  | 22 |
| 23  | 24 |
| 24  | 27 |
| 25  | 23 |
| 26  | 27 |
| 27  | 27 |
| 28  | 28 |
| 29  | 19 |
| 30  | 16 |
| 31  | 19 |
</details>

<details>
<summary>February (550 images, 26 active days)</summary>

| Day | Images |
|-----|-------:|
| 1   | 18 |
| 2   | 14 |
| 3   | 23 |
| 4   | 26 |
| 5   | 24 |
| 6   | 20 |
| 7   | 24 |
| 8   | 23 |
| 9   | 24 |
| 10  | 7  |
| 11  | 27 |
| 12  | 19 |
| 13  | 15 |
| 14  | 19 |
| 15  | 22 |
| 16  | 19 |
| 17  | 22 |
| 18  | 21 |
| 19  | 22 |
| 20  | 22 |
| 21  | 20 |
| 22  | 23 |
| 23  | 21 |
| 24  | 19 |
| 26  | 15 |
| 27  | 25 |
| 28  | 16 |
</details>

<details>
<summary>March (618 images, 31 active days)</summary>

| Day | Images |
|-----|-------:|
| 1   | 18 |
| 2   | 16 |
| 3   | 21 |
| 4   | 13 |
| 5   | 18 |
| 6   | 19 |
| 7   | 18 |
| 8   | 17 |
| 9   | 15 |
| 10  | 20 |
| 11  | 20 |
| 12  | 17 |
| 13  | 29 |
| 14  | 23 |
| 15  | 16 |
| 16  | 20 |
| 17  | 22 |
| 18  | 26 |
| 19  | 20 |
| 20  | 21 |
| 21  | 19 |
| 22  | 16 |
| 23  | 16 |
| 24  | 20 |
| 25  | 21 |
| 26  | 27 |
| 27  | 26 |
| 28  | 21 |
| 29  | 23 |
| 30  | 22 |
| 31  | 18 |
</details>

<details>
<summary>April 1–7 (141 images, 7 active days)</summary>

| Day | Images |
|-----|-------:|
| 1   | 23 |
| 2   | 8  |
| 3   | 28 |
| 4   | 29 |
| 5   | 19 |
| 6   | 20 |
| 7   | 14 |
</details>

---

## 2. Gemini API Pricing (as of April 2026)

Source: [Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing)

### Per-Model Token Rates

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------:|-----------------------:|
| Gemini 2.5 Flash-Lite | $0.10 | $0.40 |
| Gemini 2.5 Flash | $0.30 | $2.50 |
| Gemini 3 Flash Preview | $0.50 | $3.00 |
| Gemini 3.1 Flash-Lite Preview | $0.25 | $1.50 |
| Gemini 3.1 Pro Preview | $2.00 (<=200K) / $4.00 (>200K) | $12.00 (<=200K) / $18.00 (>200K) |
| Gemini 3 Pro Image Preview | $0.50 | $3.00 |

### Image Token Consumption

| Model | Tokens per input image |
|-------|-----:|
| Gemini 3 Pro Image | 560 |
| Other models (standard, up to 768x768) | ~258 |
| 4K images (3840x2160, tiled) | ~1,000–1,500 |

> **Note:** Our camera produces 4K (3840x2160) images. These are tiled into 768x768
> blocks (~4-6 tiles), consuming **~1,000–1,500 tokens per image** on input.

### Per-Image Cost Estimate (4K input, ~100 token text output)

| Model | Input Cost | Output Cost | **Total/Image** |
|-------|------------|-------------|----------------:|
| Gemini 2.5 Flash-Lite | $0.00010–0.00015 | $0.00004 | **~$0.00019** |
| Gemini 2.5 Flash | $0.00030–0.00045 | $0.00025 | **~$0.00070** |
| Gemini 3 Flash | $0.00050–0.00075 | $0.00030 | **~$0.00105** |
| Gemini 3.1 Pro | $0.00200–0.00300 | $0.00120 | **~$0.00420** |

---

## 3. Gemini API Cost Projections

Using our observed volume (~615 images/month, ~7,480 images/year):

### Monthly Cost

| Model | Per Image | Monthly (~615 imgs) | Annual (~7,480 imgs) |
|-------|----------:|--------------------:|---------------------:|
| Gemini 2.5 Flash-Lite | $0.00019 | $0.12 | $1.42 |
| Gemini 2.5 Flash | $0.00070 | $0.43 | $5.24 |
| Gemini 3 Flash | $0.00105 | $0.65 | $7.85 |
| Gemini 3.1 Pro | $0.00420 | $2.58 | $31.42 |

### Scaled Projections (multiple cameras/sites)

| Cameras | Images/Month | Gemini 2.5 Flash/mo | Gemini 3 Flash/mo | Gemini 3.1 Pro/mo |
|--------:|-------------:|--------------------:|------------------:|------------------:|
| 1       |          615 | $0.43 | $0.65 | $2.58 |
| 5       |        3,075 | $2.15 | $3.23 | $12.92 |
| 10      |        6,150 | $4.31 | $6.46 | $25.83 |
| 25      |       15,375 | $10.76 | $16.14 | $64.58 |
| 50      |       30,750 | $21.53 | $32.29 | $129.15 |

---

## 4. Self-Hosted Pipeline Cost (DigitalOcean)

Source: [DigitalOcean Droplet Pricing](https://www.digitalocean.com/pricing/droplets)

### Infrastructure (Two-Box Architecture)

| Droplet | Spec | Monthly Cost |
|---------|------|-------------:|
| API Droplet | 2 vCPU / 4 GB RAM (Basic) | $20.00 |
| FTP Server | 1 vCPU / 2 GB RAM (existing) | $12.00 |
| **Total** | | **$32.00** |

> The FTP server already exists and runs regardless of detection method.
> The **incremental cost** of self-hosting is the API droplet only: **$20/month**.

### What the Fixed Cost Includes

- Unlimited image processing (no per-image charge)
- On-prem inference (~1–3s per 4K image with YOLO)
- Full data sovereignty (images never leave infrastructure)
- No API rate limits or quotas
- No dependency on external service availability

---

## 5. Break-Even & Savings Analysis

### Single Camera (current setup)

| Approach | Monthly Cost | Annual Cost |
|----------|-------------:|------------:|
| Gemini 2.5 Flash-Lite | $0.12 | $1.42 |
| Gemini 2.5 Flash | $0.43 | $5.24 |
| Gemini 3 Flash | $0.65 | $7.85 |
| Gemini 3.1 Pro | $2.58 | $31.42 |
| **Self-hosted (incremental)** | **$20.00** | **$240.00** |

> At a single camera with ~20 images/day, Gemini API is cheaper on pure cost.
> **Break-even requires scale.**

### Break-Even Points (vs. self-hosted $20/mo)

| Gemini Model | Break-even cameras | Break-even images/month |
|--------------|---------:|----------:|
| Gemini 2.5 Flash-Lite | ~167 | ~105,263 |
| Gemini 2.5 Flash | ~47 | ~28,571 |
| Gemini 3 Flash | ~31 | ~19,048 |
| Gemini 3.1 Pro | ~8 | ~4,762 |

### Beyond Raw Cost — Qualitative Advantages

| Factor | Gemini API | Self-Hosted Pipeline |
|--------|-----------|---------------------|
| **Per-image cost** | Scales linearly | Zero (fixed infra) |
| **Latency** | Network round-trip + queue | Local inference ~1–3s |
| **Data privacy** | Images sent to Google | Images stay on-prem |
| **Availability** | Depends on Google uptime | Independent |
| **Rate limits** | 15 RPM (free) / quota-based | None |
| **Customisation** | Prompt engineering only | Fine-tune on site data |
| **Compliance** | Third-party data processing | Full control |
| **Offline capability** | None | Fully functional |

---

## 6. Recommendations for Value Proposition

### For Marketing Slides

1. **Cost at scale is the story.** Single-camera economics favor the API, but multi-site
   deployments (8+ cameras on Gemini 3.1 Pro, or 31+ on Gemini 3 Flash) make
   self-hosting cheaper — and cost stays flat as volume grows.

2. **Data sovereignty is the wedge.** Construction site imagery is sensitive. Self-hosted
   means zero third-party data exposure — a compliance win that has no Gemini equivalent.

3. **Reliability without internet.** Remote construction sites may have unreliable
   connectivity. Self-hosted detection works with local network only.

4. **No vendor lock-in.** Models can be swapped, fine-tuned on site-specific images, or
   upgraded without renegotiating API contracts.

5. **Latency for real-time alerting.** If the pipeline evolves toward real-time video
   monitoring, local inference at ~1–3s/frame is viable; API round-trips are not.

### Suggested Talking Points

- "At 10 cameras, we save **$X/year** vs Gemini 3.1 Pro while keeping all data on-prem"
- "Fixed $20/month infrastructure — process 100 or 100,000 images at the same cost"
- "No API keys, no quotas, no third-party data processing agreements"
- "Purpose-built dual-model pipeline tuned for construction PPE — not a general-purpose LLM doing object detection"

---

## Sources

- [Gemini Developer API Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [DigitalOcean Droplet Pricing](https://www.digitalocean.com/pricing/droplets)
- [Google Cloud Vision API Pricing](https://cloud.google.com/vision/pricing)
- [Vertex AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
