"""
AstraShield | tests/test_api.py
Integration tests for the FastAPI REST server.
Uses TestClient — no real server needed.
"""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from fastapi.testclient import TestClient
    _FASTAPI = True
except ImportError:
    _FASTAPI = False

pytestmark = pytest.mark.skipif(not _FASTAPI, reason="fastapi not installed")


@pytest.fixture(scope="module")
def client():
    from api.server import app
    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_system_field(self, client):
        data = client.get("/").json()
        assert data["system"] == "AstraShield"

    def test_root_has_tagline(self, client):
        data = client.get("/").json()
        assert "Predict" in data["tagline"]


class TestStatusEndpoint:
    def test_status_200(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200

    def test_status_fields(self, client):
        data = client.get("/api/status").json()
        for field in ["system","sim_time","satellites","debris_tracked"]:
            assert field in data

    def test_satellites_count(self, client):
        data = client.get("/api/status").json()
        assert data["satellites"] == 50


class TestTelemetryEndpoint:
    def test_telemetry_ack(self, client):
        payload = {
            "timestamp": "2026-03-12T08:00:00.000Z",
            "objects": [{
                "id": "DEB-TEST-001",
                "type": "DEBRIS",
                "r": {"x": 4500.2, "y": -2100.5, "z": 4800.1},
                "v": {"x": -1.25,  "y": 6.84,    "z": 3.12}
            }]
        }
        resp = client.post("/api/telemetry", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ACK"
        assert data["processed_count"] == 1

    def test_telemetry_multiple_objects(self, client):
        payload = {
            "timestamp": "2026-03-12T08:00:00.000Z",
            "objects": [
                {"id": f"DEB-BATCH-{i}", "type": "DEBRIS",
                 "r": {"x": 4500.+i, "y": -2100., "z": 4800.},
                 "v": {"x": -1.2, "y": 6.8, "z": 3.1}}
                for i in range(5)
            ]
        }
        resp = client.post("/api/telemetry", json=payload)
        assert resp.status_code == 200
        assert resp.json()["processed_count"] == 5


class TestManeuverEndpoint:
    def test_schedule_valid_satellite(self, client):
        payload = {
            "satelliteId": "SAT-00-00",
            "maneuver_sequence": [{
                "burn_id": "EVASION_BURN_1",
                "burnTime": "2026-03-12T14:15:30.000Z",
                "deltaV_vector": {"x": 0.002, "y": 0.015, "z": -0.001}
            }]
        }
        resp = client.post("/api/maneuver/schedule", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SCHEDULED"
        assert "validation" in data
        assert "projected_mass_remaining_kg" in data["validation"]

    def test_schedule_unknown_satellite_404(self, client):
        payload = {
            "satelliteId": "SAT-NONEXISTENT-99",
            "maneuver_sequence": []
        }
        resp = client.post("/api/maneuver/schedule", json=payload)
        assert resp.status_code == 404


class TestSimulateStep:
    def test_step_completes(self, client):
        resp = client.post("/api/simulate/step", json={"step_seconds": 60.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "STEP_COMPLETE"
        assert "new_timestamp" in data

    def test_step_increments_time(self, client):
        t1 = client.get("/api/status").json()["sim_time"]
        client.post("/api/simulate/step", json={"step_seconds": 3600.0})
        t2 = client.get("/api/status").json()["sim_time"]
        assert t2 > t1


class TestSnapshotEndpoint:
    def test_snapshot_200(self, client):
        resp = client.get("/api/visualization/snapshot")
        assert resp.status_code == 200

    def test_snapshot_has_satellites(self, client):
        data = client.get("/api/visualization/snapshot").json()
        assert "satellites" in data
        assert len(data["satellites"]) > 0

    def test_snapshot_satellite_fields(self, client):
        data = client.get("/api/visualization/snapshot").json()
        sat = data["satellites"][0]
        for f in ["id","lat","lon","fuel_kg","status"]:
            assert f in sat

    def test_snapshot_lat_lon_range(self, client):
        data = client.get("/api/visualization/snapshot").json()
        for sat in data["satellites"]:
            assert -90. <= sat["lat"] <= 90.
            assert -180. <= sat["lon"] <= 180.

    def test_snapshot_has_debris_cloud(self, client):
        data = client.get("/api/visualization/snapshot").json()
        assert "debris_cloud" in data
