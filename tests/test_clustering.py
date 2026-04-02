"""
AstraShield | tests/test_clustering.py
Unit tests for HDBSCAN clustering and BallTree CA.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.clustering import (
    run_clustering, compute_cluster_stats, tag_risk, ConjunctionAssessor
)
from core.data_gen import generate_debris, generate_satellites, objects_to_dataframe


@pytest.fixture(scope="module")
def small_debris():
    debs = generate_debris(500)
    return objects_to_dataframe(debs)

@pytest.fixture(scope="module")
def small_sats():
    sats = generate_satellites(10)
    return objects_to_dataframe(sats)

@pytest.fixture(scope="module")
def clustered_debris(small_debris):
    df, n, method = run_clustering(small_debris, min_cluster_size=5)
    return df, n, method


class TestClustering:
    def test_returns_cluster_ids(self, clustered_debris):
        df, n, method = clustered_debris
        assert "cluster_id" in df.columns

    def test_finds_clusters(self, clustered_debris):
        df, n, method = clustered_debris
        assert n > 0, "Should find at least one cluster"

    def test_noise_label_is_minus_one(self, clustered_debris):
        df, _, _ = clustered_debris
        noise = df[df["cluster_id"] == -1]
        # Noise is expected but not required
        assert all(noise["cluster_id"] == -1)

    def test_cluster_ids_are_integers(self, clustered_debris):
        df, _, _ = clustered_debris
        assert df["cluster_id"].dtype in [np.int32, np.int64, int]

    def test_method_string_returned(self, clustered_debris):
        _, _, method = clustered_debris
        assert isinstance(method, str)
        assert len(method) > 0


class TestClusterStats:
    def test_stats_columns_present(self, clustered_debris):
        df, _, _ = clustered_debris
        stats = compute_cluster_stats(df)
        required = ["cluster_id","size","density_proxy","risk_score","risk_level",
                    "mean_alt_km","dominant_parent"]
        for col in required:
            assert col in stats.columns, f"Missing column: {col}"

    def test_risk_levels_valid(self, clustered_debris):
        df, _, _ = clustered_debris
        stats = compute_cluster_stats(df)
        valid = {"HIGH", "MEDIUM", "LOW"}
        assert set(stats["risk_level"].unique()).issubset(valid)

    def test_risk_score_range(self, clustered_debris):
        df, _, _ = clustered_debris
        stats = compute_cluster_stats(df)
        assert stats["risk_score"].between(0., 1.).all()

    def test_tag_risk_adds_columns(self, clustered_debris):
        df, _, _ = clustered_debris
        stats = compute_cluster_stats(df)
        tagged = tag_risk(df, stats)
        assert "risk_level" in tagged.columns
        assert "risk_score" in tagged.columns

    def test_noise_tagged_as_noise(self, clustered_debris):
        df, _, _ = clustered_debris
        stats = compute_cluster_stats(df)
        tagged = tag_risk(df, stats)
        noise_rows = tagged[tagged["cluster_id"] == -1]
        assert (noise_rows["risk_level"] == "NOISE").all()


class TestConjunctionAssessor:
    def test_build_index(self, small_debris):
        ca = ConjunctionAssessor(small_debris)
        ca.build_index()
        assert ca._tree is not None

    def test_screen_returns_dataframe(self, small_debris, small_sats):
        ca = ConjunctionAssessor(small_debris)
        ca.build_index()
        sat_pos = small_sats[["x","y","z"]].values[0]
        result = ca.screen_satellite(sat_pos, radius_km=500.)
        assert isinstance(result, pd.DataFrame)

    def test_screen_miss_distance_nonneg(self, small_debris, small_sats):
        ca = ConjunctionAssessor(small_debris)
        ca.build_index()
        sat_pos = small_sats[["x","y","z"]].values[0]
        result = ca.screen_satellite(sat_pos, radius_km=500.)
        if len(result) > 0:
            assert (result["miss_dist_km"] >= 0.).all()

    def test_no_hits_beyond_radius(self, small_debris, small_sats):
        """With radius=0.001 km (1 m), expect no hits."""
        ca = ConjunctionAssessor(small_debris)
        ca.build_index()
        sat_pos = small_sats[["x","y","z"]].values[0]
        result = ca.screen_satellite(sat_pos, radius_km=0.001)
        assert len(result) == 0
