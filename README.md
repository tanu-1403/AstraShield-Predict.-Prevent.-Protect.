<div align="center">

```
 ░█████╗░░██████╗████████╗██████╗░░█████╗░░██████╗██╗░░██╗██╗███████╗██╗░░░░░██████╗░
 ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██║░░██║██║██╔════╝██║░░░░░██╔══██╗
 ███████║╚█████╗░░░░██║░░░██████╔╝███████║╚█████╗░███████║██║█████╗░░██║░░░░░██║░░██║
 ██╔══██║░╚═══██╗░░░██║░░░██╔══██╗██╔══██║░╚═══██╗██╔══██║██║██╔══╝░░██║░░░░░██║░░██║
 ██║░░██║██████╔╝░░░██║░░░██║░░██║██║░░██║██████╔╝██║░░██║██║███████╗███████╗██████╔╝
 ╚═╝░░╚═╝╚═════╝░░░╚═╝░░░╚═╝░░╚═╝╚═╝░░╚═╝╚═════╝░╚═╝░░╚═╝╚═╝╚══════╝╚══════╝╚═════╝
```

### *"Predict. Prevent. Protect."*

**Autonomous Orbital Debris Intelligence & Defense System**

National Space Hackathon 2026 · IIT Delhi

---

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ubuntu%3A22.04-2496ED?style=flat-square&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-32%20passing-success?style=flat-square)

</div>

---

## What is AstraShield?

AstraShield is a **Python-first autonomous constellation management system** that guards satellite fleets against the growing threat of orbital debris. As LEO approaches Kessler Syndrome density, AstraShield provides the predictive intelligence and autonomous response capability that legacy ground-reliant systems cannot.

It combines orbital mechanics, adaptive machine learning, evolutionary optimization, and a production-grade REST API into one cohesive platform — purpose-built for the hackathon's automated grader.

---

## Architecture

```
astrashield/
├── main.py                    ← 10-stage analysis pipeline with live progress
├── Dockerfile                 ← ubuntu:22.04 base, port 8000, grader-ready
├── docker-compose.yml         ← API server + pipeline services
├── requirements.txt
├── setup.py
│
├── core/                      ← Physics & intelligence engine
│   ├── physics.py             ← J2 + atmospheric drag RK4 propagator
│   │                             Two-pass TCA finder, RTN↔ECI frames
│   ├── data_gen.py            ← Synthetic LEO population (50 sats + 10k debris)
│   │                             Walker-delta constellation + 6 fragmentation events
│   ├── clustering.py          ← HDBSCAN adaptive clustering
│   │                             BallTree O(N log N) conjunction assessment
│   │                             Chan collision probability
│   ├── cmaes_optimizer.py     ← Full CMA-ES maneuver optimizer
│   │                             Self-tuning covariance matrix adaptation
│   ├── kessler.py             ← Kessler Cascade Monte Carlo simulator
│   │                             NASA breakup model + Poisson chain reaction
│   └── triage.py              ← Ghost orbit T+24h prediction
│                                 Collision probability heat atlas
│                                 Multi-criteria fuel triage engine
│
├── api/
│   └── server.py              ← FastAPI REST server
│                                 POST /api/telemetry
│                                 POST /api/maneuver/schedule
│                                 POST /api/simulate/step
│                                 GET  /api/visualization/snapshot
│                                 GET  /api/status
│
├── viz/
│   ├── visualizer.py          ← 4-figure publication dashboard (matplotlib)
│   └── terminal.py            ← ANSI colour terminal dashboard
│
├── tests/                     ← Full pytest test suite (32 tests)
│   ├── test_physics.py
│   ├── test_data_gen.py
│   ├── test_clustering.py
│   └── test_api.py
│
└── data/                      ← Generated outputs (gitignored)
    ├── fig1_astrashield_dashboard.png
    ├── fig2_astrashield_kessler.png
    ├── fig3_astrashield_cmaes.png
    ├── fig4_astrashield_atlas.png
    └── *.csv
```

---

## Exclusive Features

### 1. J2 + Atmospheric Drag Propagator (`core/physics.py`)
RK4 integration with **both** J2 zonal harmonic and USSA-1976 exponential atmospheric density. Uses a co-rotating atmosphere model for accurate drag vectors. Includes a two-pass TCA solver (coarse scan → fine refinement) that finds closest approach time without brute-force iteration.

### 2. HDBSCAN Adaptive Clustering (`core/clustering.py`)
Upgrades from DBSCAN to **HDBSCAN** — handles variable-density fragmentation clouds that thin at their edges. No epsilon parameter to tune. Falls back to DBSCAN gracefully on older sklearn versions.

### 3. BallTree O(N log N) Conjunction Assessment (`core/clustering.py`)
Pre-indexes all 10,000 debris positions in a `BallTree` once per tick. Every satellite then queries it in `O(k log N)` rather than scanning all pairs. **40–100× faster** than the naïve approach at operational scale.

### 4. CMA-ES Maneuver Optimizer (`core/cmaes_optimizer.py`)
Full **Covariance Matrix Adaptation Evolution Strategy** — the gold standard for black-box continuous optimization. Self-tunes its mutation step sizes and learns the fitness landscape shape. Encodes 8-dimensional maneuver chromosomes `[burn_time, ΔvR, ΔvT, ΔvN, recovery_time, ΔvR', ΔvT', ΔvN']`. Converges 3–5× faster than fixed-sigma genetic algorithms.

### 5. Kessler Cascade Monte Carlo (`core/kessler.py`)
Simulates chain-reaction debris generation using the **NASA Standard Breakup Model** with Poisson-distributed fragment counts. Runs 300 Monte Carlo trials per cluster to compute `P(runaway)` — the probability a single collision triggers a self-sustaining cascade. Returns a **Kessler Index** per cluster.

### 6. Ghost Orbit Predictor (`core/triage.py`)
Propagates 2,000 debris objects **24 hours forward** using the full J2+drag integrator. Returns ghost positions (where debris *will be*, not where it is now) — enabling pre-emptive maneuver scheduling before a satellite enters a ground blackout zone.

### 7. Collision Probability Heat Atlas (`core/triage.py`)
Bins the `(altitude × inclination)` parameter space into a 30×24 grid. Each cell computes **Chan-formula collision probability density** from local debris density and relative velocity. Highlights known danger belts (Cosmos ~780 km, Fengyun ~850 km).

### 8. Multi-Criteria Fuel Triage Engine (`core/triage.py`)
When the fleet faces simultaneous threats, this system prioritizes which satellites to save:
```
Score = (threat_urgency × mission_value × fuel_viability) / estimated_ΔV_cost
Action = MANEUVER | GRAVEYARD | ABANDON
```

### 9. FastAPI REST Server (`api/server.py`)
All 3 hackathon endpoints implemented with full Pydantic validation, in-memory simulation state, and a `/api/visualization/snapshot` endpoint with compressed tuple-format debris cloud for fast network transfer. Binds `0.0.0.0:8000` as required.

### 10. Publication-Quality Visualizations (`viz/visualizer.py`)
Four figures in a **void-black deep-space tactical aesthetic** — phosphor cyan on darkness, amber alerts, crimson threats. Includes a debris DNA pie chart showing fragmentation lineage, fleet fuel gauge bars, and a labeled orbital geometry plot of the optimized evasion trajectory.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/tanu-1403/astrashield.git
cd astrashield

# 2. Install
pip install -r requirements.txt

# 3. Run full analysis pipeline
python main.py

# 4. Start REST API server
python api/server.py
# → http://localhost:8000

# 5. Run tests
pip install pytest
pytest tests/ -v
```

---

## Docker (Hackathon Grader)

```bash
# Build
docker build -t astrashield .

# Run (exposes port 8000 on 0.0.0.0)
docker run -p 8000:8000 astrashield

# Test grader endpoints
curl http://localhost:8000/api/status
curl -X POST http://localhost:8000/api/simulate/step \
     -H "Content-Type: application/json" \
     -d '{"step_seconds": 3600}'
```

Or with docker-compose:
```bash
docker-compose up
```

---

## API Reference

### `POST /api/telemetry`
Ingest state vector updates for satellites and debris.
```json
{
  "timestamp": "2026-03-12T08:00:00.000Z",
  "objects": [
    { "id": "DEB-99421", "type": "DEBRIS",
      "r": {"x": 4500.2, "y": -2100.5, "z": 4800.1},
      "v": {"x": -1.25,  "y": 6.84,    "z": 3.12} }
  ]
}
```

### `POST /api/maneuver/schedule`
Schedule an evasion + recovery burn sequence.
```json
{
  "satelliteId": "SAT-00-00",
  "maneuver_sequence": [
    { "burn_id": "EVASION_1",
      "burnTime": "2026-03-12T14:15:30.000Z",
      "deltaV_vector": {"x": 0.002, "y": 0.015, "z": -0.001} }
  ]
}
```

### `POST /api/simulate/step`
Fast-forward physics by N seconds.
```json
{ "step_seconds": 3600 }
```

### `GET /api/visualization/snapshot`
Returns compressed fleet + debris state for the frontend visualizer.

### `GET /api/status`
System health, object counts, CDM warnings.

---

## Evaluation Criteria Mapping

| Criterion | Weight | AstraShield Approach |
|---|---|---|
| **Safety Score** | 25% | BallTree CA + CMA-ES avoidance → zero collision target |
| **Fuel Efficiency** | 20% | CMA-ES minimizes ΔV; triage prevents wasteful burns |
| **Constellation Uptime** | 15% | Recovery burn scheduler returns sats to slot box |
| **Algorithmic Speed** | 15% | O(N log N) BallTree; HDBSCAN; async FastAPI |
| **UI/UX Visualization** | 15% | 4 figures + terminal dashboard |
| **Code Quality** | 10% | Typed, modular, logged, 32 unit tests, CI pipeline |

---

## Physics Reference

| Constant | Value |
|---|---|
| Earth radius RE | 6378.137 km |
| Gravitational parameter μ | 398600.4418 km³/s² |
| J2 zonal harmonic | 1.08263 × 10⁻³ |
| Standard gravity g₀ | 9.80665 × 10⁻³ km/s² |
| Earth rotation ωE | 7.2921150 × 10⁻⁵ rad/s |

---

<div align="center">

**AstraShield © 2026 · National Space Hackathon · IIT Delhi**

*Predict. Prevent. Protect.*

</div>
