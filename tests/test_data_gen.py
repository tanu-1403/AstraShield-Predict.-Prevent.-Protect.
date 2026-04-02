"""
AstraShield | tests/test_data_gen.py
Unit tests for synthetic orbital population generator.
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.data_gen import generate_satellites, generate_debris, objects_to_dataframe
from core.physics import RE


class TestGenerateSatellites:
    def test_count(self):
        sats = generate_satellites(50)
        assert len(sats) == 50

    def test_all_satellite_type(self):
        sats = generate_satellites(10)
        assert all(s.obj_type == "SATELLITE" for s in sats)

    def test_unique_ids(self):
        sats = generate_satellites(50)
        ids = [s.obj_id for s in sats]
        assert len(set(ids)) == 50

    def test_altitude_range(self):
        sats = generate_satellites(50)
        df = objects_to_dataframe(sats)
        assert df["altitude_km"].between(400., 1000.).all()

    def test_speed_range(self):
        sats = generate_satellites(50)
        df = objects_to_dataframe(sats)
        assert df["speed_kms"].between(6.5, 8.5).all()

    def test_fuel_positive(self):
        sats = generate_satellites(50)
        df = objects_to_dataframe(sats)
        assert (df["fuel_kg"] > 0.).all()


class TestGenerateDebris:
    def test_count(self):
        debs = generate_debris(1000)
        assert len(debs) == 1000

    def test_all_debris_type(self):
        debs = generate_debris(100)
        assert all(d.obj_type == "DEBRIS" for d in debs)

    def test_altitude_range(self):
        debs = generate_debris(500)
        df = objects_to_dataframe(debs)
        assert df["altitude_km"].between(150., 2100.).all()

    def test_speed_realistic(self):
        debs = generate_debris(500)
        df = objects_to_dataframe(debs)
        # All debris should be in orbital speed range (some spread allowed for fragments)
        assert df["speed_kms"].between(5.5, 10.0).all()

    def test_parent_event_populated(self):
        debs = generate_debris(500)
        df = objects_to_dataframe(debs)
        assert "parent_event" in df.columns
        assert df["parent_event"].notna().all()

    def test_frag_events_present(self):
        debs = generate_debris(1000)
        df = objects_to_dataframe(debs)
        # Some debris should come from named fragmentation events
        named = df[df["parent_event"] == "COSMOS-BREAKUP"]
        assert len(named) > 0


class TestDataFrame:
    def test_required_columns(self):
        sats = generate_satellites(5)
        df = objects_to_dataframe(sats)
        required = ["id","type","x","y","z","vx","vy","vz",
                    "altitude_km","speed_kms","fuel_kg"]
        for col in required:
            assert col in df.columns

    def test_no_nan_positions(self):
        debs = generate_debris(200)
        df = objects_to_dataframe(debs)
        assert not df[["x","y","z"]].isna().any().any()

    def test_no_nan_velocities(self):
        debs = generate_debris(200)
        df = objects_to_dataframe(debs)
        assert not df[["vx","vy","vz"]].isna().any().any()
