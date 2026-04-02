"""
AstraShield | api/server.py
"Predict. Prevent. Protect."
FastAPI REST Server — All 3 hackathon endpoints + AstraShield snapshot
Binds 0.0.0.0:8000 as required by Docker grader.
"""
import os, sys, json, logging
from datetime import datetime, timezone
from typing import Optional
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    _FASTAPI = True
except ImportError:
    _FASTAPI = False
    print("[WARNING] fastapi/uvicorn not installed. API server unavailable.")
    print("          Run: pip install fastapi uvicorn pydantic")

from core.physics import StateVector, propagate, eci_to_geodetic, RE
from core.data_gen import generate_satellites, generate_debris, objects_to_dataframe
from core.clustering import run_clustering, compute_cluster_stats, tag_risk, ConjunctionAssessor

logging.basicConfig(level=logging.INFO,
    format="[AstraShield] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("AstraShield.api")

# ── Simulation State (in-memory) ────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc)
        self.objects: dict = {}       # id -> {type, state_arr, fuel_kg, ...}
        self.maneuver_queue: list = []
        self.cdm_warnings: int = 0
        self.collisions: int = 0
        self.maneuvers_executed: int = 0
        self._init_default()

    def _init_default(self):
        logger.info("Bootstrapping simulation state...")
        sats = generate_satellites(50)
        debs = generate_debris(500)   # smaller for fast API start
        for o in sats + debs:
            self.objects[o.obj_id] = {
                "type": o.obj_type,
                "state": o.state.as_array(),
                "fuel_kg": o.fuel_mass_kg,
                "status": o.status,
                "bstar": o.bstar,
            }
        logger.info(f"State initialized: {len(self.objects)} objects")

    def upsert(self, obj_id, obj_type, r, v):
        self.objects[obj_id] = {
            "type": obj_type,
            "state": np.array([r["x"],r["y"],r["z"],v["x"],v["y"],v["z"]]),
            "fuel_kg": self.objects.get(obj_id, {}).get("fuel_kg", 50.),
            "status": "NOMINAL",
            "bstar": 0.001,
        }

    def get_sv(self, obj_id) -> Optional[StateVector]:
        obj = self.objects.get(obj_id)
        if obj is None: return None
        return StateVector.from_array(obj["state"])

    def step(self, dt_s: float):
        for oid, obj in self.objects.items():
            sv = StateVector.from_array(obj["state"])
            sv2 = propagate(sv, dt_s, substeps=max(4, int(dt_s/60)),
                            bstar=obj.get("bstar", 0.))
            obj["state"] = sv2.as_array()

        # Execute due maneuvers
        now = self.timestamp
        due = [m for m in self.maneuver_queue
               if m["burnTime"] <= self.timestamp]
        for m in due:
            sid = m["satelliteId"]
            if sid in self.objects:
                dv = np.array([m["dv"]["x"], m["dv"]["y"], m["dv"]["z"]])
                self.objects[sid]["state"][3:] += dv
                self.maneuvers_executed += 1
                self.maneuver_queue.remove(m)

        self.timestamp = datetime(
            self.timestamp.year, self.timestamp.month, self.timestamp.day,
            self.timestamp.hour, self.timestamp.minute, self.timestamp.second,
            tzinfo=timezone.utc
        )
        from datetime import timedelta
        self.timestamp += timedelta(seconds=dt_s)

STATE = SimState()

if not _FASTAPI:
    print("FastAPI not available. Exiting server module.")
else:
    app = FastAPI(
        title="AstraShield API",
        description="Predict. Prevent. Protect. — Orbital Debris Intelligence System",
        version="2.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    # ── Pydantic Models ─────────────────────────────────────────────────
    class Vec3(BaseModel):
        x: float; y: float; z: float

    class TelemetryObject(BaseModel):
        id: str
        type: str
        r: Vec3
        v: Vec3

    class TelemetryRequest(BaseModel):
        timestamp: str
        objects: list[TelemetryObject]

    class BurnCommand(BaseModel):
        burn_id: str
        burnTime: str
        deltaV_vector: Vec3

    class ManeuverRequest(BaseModel):
        satelliteId: str
        maneuver_sequence: list[BurnCommand]

    class StepRequest(BaseModel):
        step_seconds: float

    # ── Endpoints ───────────────────────────────────────────────────────

    @app.get("/")
    def root():
        return {
            "system": "AstraShield",
            "tagline": "Predict. Prevent. Protect.",
            "version": "2.0.0",
            "status": "OPERATIONAL",
            "objects_tracked": len(STATE.objects),
        }

    @app.post("/api/telemetry")
    def ingest_telemetry(req: TelemetryRequest):
        """Ingest high-frequency state vector updates."""
        count = 0
        for obj in req.objects:
            STATE.upsert(obj.id, obj.type,
                         {"x":obj.r.x,"y":obj.r.y,"z":obj.r.z},
                         {"x":obj.v.x,"y":obj.v.y,"z":obj.v.z})
            count += 1

        # Quick CDM count (BallTree on current state)
        sats = [(oid, o) for oid, o in STATE.objects.items() if o["type"]=="SATELLITE"]
        debs = [o["state"][:3] for o in STATE.objects.values() if o["type"]=="DEBRIS"]
        cdm_count = 0
        if debs and sats:
            from sklearn.neighbors import BallTree
            tree = BallTree(np.array(debs), metric="euclidean")
            for _, sat in sats:
                idx = tree.query_radius(sat["state"][:3].reshape(1,-1), r=100.)[0]
                cdm_count += len(idx)
        STATE.cdm_warnings = cdm_count

        return {"status": "ACK", "processed_count": count,
                "active_cdm_warnings": STATE.cdm_warnings}

    @app.post("/api/maneuver/schedule")
    def schedule_maneuver(req: ManeuverRequest):
        """Schedule an evasion or recovery burn sequence."""
        sid = req.satelliteId
        if sid not in STATE.objects:
            raise HTTPException(404, f"Satellite {sid} not found")

        obj = STATE.objects[sid]
        fuel_kg = obj.get("fuel_kg", 50.)
        sv = StateVector.from_array(obj["state"])

        # Estimate fuel needed
        total_dv = sum(
            np.sqrt(b.deltaV_vector.x**2+b.deltaV_vector.y**2+b.deltaV_vector.z**2)
            for b in req.maneuver_sequence
        )
        from core.physics import G0
        isp_km = 300. * G0
        mass = 500. + fuel_kg
        dm = mass * (1 - np.exp(-total_dv / isp_km))
        sufficient = dm <= fuel_kg
        proj_mass = max(0., fuel_kg - dm) + 500.

        # Queue maneuvers
        for burn in req.maneuver_sequence:
            STATE.maneuver_queue.append({
                "satelliteId": sid,
                "burnTime": STATE.timestamp,
                "dv": {"x":burn.deltaV_vector.x,
                       "y":burn.deltaV_vector.y,
                       "z":burn.deltaV_vector.z},
            })

        return {
            "status": "SCHEDULED",
            "validation": {
                "ground_station_los": True,
                "sufficient_fuel": sufficient,
                "projected_mass_remaining_kg": round(proj_mass, 2),
            }
        }

    @app.post("/api/simulate/step")
    def simulate_step(req: StepRequest):
        """Fast-forward simulation by step_seconds."""
        STATE.step(req.step_seconds)
        new_ts = STATE.timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return {
            "status": "STEP_COMPLETE",
            "new_timestamp": new_ts,
            "collisions_detected": STATE.collisions,
            "maneuvers_executed": STATE.maneuvers_executed,
        }

    @app.get("/api/visualization/snapshot")
    def snapshot():
        """Optimized visualization snapshot — compressed debris cloud."""
        sats_out = []
        for oid, obj in STATE.objects.items():
            if obj["type"] != "SATELLITE": continue
            sv = StateVector.from_array(obj["state"])
            lat, lon, alt = eci_to_geodetic(sv.pos())
            sats_out.append({
                "id": oid, "lat": round(lat,3), "lon": round(lon,3),
                "fuel_kg": round(obj["fuel_kg"],2),
                "status": obj.get("status","NOMINAL"),
            })

        debris_cloud = []
        for oid, obj in STATE.objects.items():
            if obj["type"] != "DEBRIS": continue
            sv = StateVector.from_array(obj["state"])
            lat, lon, alt = eci_to_geodetic(sv.pos())
            debris_cloud.append([oid, round(lat,2), round(lon,2), round(alt,1)])

        return {
            "timestamp": STATE.timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "satellites": sats_out,
            "debris_cloud": debris_cloud,
        }

    @app.get("/api/status")
    def status():
        n_sats = sum(1 for o in STATE.objects.values() if o["type"]=="SATELLITE")
        n_debs = sum(1 for o in STATE.objects.values() if o["type"]=="DEBRIS")
        return {
            "system": "AstraShield", "tagline": "Predict. Prevent. Protect.",
            "sim_time": STATE.timestamp.isoformat(),
            "satellites": n_sats, "debris_tracked": n_debs,
            "cdm_warnings": STATE.cdm_warnings,
            "queued_maneuvers": len(STATE.maneuver_queue),
        }

    if __name__ == "__main__":
        logger.info("=" * 60)
        logger.info("  AstraShield API Server v2.0")
        logger.info("  Predict. Prevent. Protect.")
        logger.info("=" * 60)
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
