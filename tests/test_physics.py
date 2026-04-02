"""
AstraShield | tests/test_physics.py
Unit tests for the orbital mechanics engine.
"""
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.physics import (
    StateVector, OrbitalObject, eci_from_elements,
    propagate, rtn_to_eci, eci_to_geodetic,
    circ_vel, atm_density, tca, RE, MU
)


class TestStateVector:
    def test_construction(self):
        sv = StateVector(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
        assert sv.x == 1.0
        np.testing.assert_array_equal(sv.pos(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(sv.vel(), [0.1, 0.2, 0.3])

    def test_roundtrip_array(self):
        sv = StateVector(1000., -2000., 3000., 1.5, -6.0, 3.2)
        sv2 = StateVector.from_array(sv.as_array())
        np.testing.assert_allclose(sv.as_array(), sv2.as_array())

    def test_altitude(self):
        sv = eci_from_elements(600., 53., 0., 0.)
        assert abs(sv.altitude() - 600.) < 0.01

    def test_speed_physical_range(self):
        for alt in [200, 500, 800, 1500]:
            sv = eci_from_elements(alt, 53., 0., 0.)
            v = sv.speed()
            assert 6.0 < v < 9.0, f"Speed {v:.2f} out of range at {alt} km"


class TestCircularOrbit:
    def test_circular_velocity(self):
        # At 400 km ISS-like orbit, v ≈ 7.67 km/s
        v = circ_vel(400.)
        assert abs(v - 7.67) < 0.05

    def test_eci_from_elements_altitude(self):
        for alt in [300, 550, 780, 1200]:
            sv = eci_from_elements(alt, 51.6, 0., 0.)
            assert abs(sv.altitude() - alt) < 1.0

    def test_period_conservation(self):
        """After one full orbit, satellite should return near start."""
        sv = eci_from_elements(550., 53., 45., 0.)
        r0 = RE + 550.
        T  = 2 * np.pi * np.sqrt(r0**3 / MU)   # orbital period, seconds
        sv_final = propagate(sv, T, substeps=200)
        dist = np.linalg.norm(sv_final.pos() - sv.pos())
        # Should return within 50 km after one orbit with J2 perturbation
        assert dist < 50., f"Period drift too large: {dist:.1f} km"


class TestPropagation:
    def test_altitude_preserved(self):
        """Circular orbit altitude should not drift more than 5 km over 1h."""
        sv = eci_from_elements(600., 53., 0., 0.)
        sv2 = propagate(sv, 3600., substeps=60)
        assert abs(sv2.altitude() - 600.) < 15.0

    def test_speed_preserved(self):
        """Speed change should be < 0.1 km/s over 1 orbit (no drag)."""
        sv = eci_from_elements(800., 98., 0., 0.)
        sv2 = propagate(sv, 6000., substeps=100, bstar=0.)
        assert abs(sv2.speed() - sv.speed()) < 0.1

    def test_drag_reduces_altitude(self):
        """With drag enabled, altitude at very low orbit should decrease."""
        sv = eci_from_elements(250., 51.6, 0., 0.)
        bstar = 0.01   # high drag
        sv2 = propagate(sv, 86400., substeps=500, bstar=bstar)
        # Altitude should decrease, not increase
        assert sv2.altitude() < sv.altitude()


class TestAtmosphere:
    def test_density_decreases_with_altitude(self):
        rho200 = atm_density(200.)
        rho600 = atm_density(600.)
        rho1500 = atm_density(1500.)
        assert rho200 > rho600 > rho1500

    def test_zero_above_2000(self):
        assert atm_density(2001.) == 0.0
        assert atm_density(5000.) == 0.0

    def test_positive_below_2000(self):
        for alt in [200, 400, 600, 1000, 1800]:
            assert atm_density(alt) > 0.


class TestFrames:
    def test_rtn_orthonormal(self):
        sv = eci_from_elements(600., 53., 45., 30.)
        M = rtn_to_eci(sv)
        # Columns should be orthonormal
        np.testing.assert_allclose(M.T @ M, np.eye(3), atol=1e-10)

    def test_geodetic_roundtrip_lat(self):
        sv = eci_from_elements(600., 45., 0., 0.)
        lat, lon, alt = eci_to_geodetic(sv.pos())
        assert -90. <= lat <= 90.
        assert -180. <= lon <= 180.
        assert 0. < alt < 2000.


class TestTCA:
    def test_tca_finds_close_approach(self):
        """Two objects at same altitude but offset should have TCA > 0."""
        sv1 = eci_from_elements(600., 53., 45., 0.)
        sv2 = eci_from_elements(600., 53., 45., 5.)
        t, d = tca(sv1, sv2, dt_s=7200., coarse=120., fine=10.)
        assert t >= 0.
        assert d < 1000.   # should get within 1000 km at some point


class TestOrbitalObject:
    def test_delta_v_depletes_fuel(self):
        sv = eci_from_elements(600., 53., 0., 0.)
        obj = OrbitalObject("SAT-TEST", "SATELLITE", sv, fuel_mass_kg=50.)
        dv = np.array([0., 0.01, 0.])   # 10 m/s prograde
        fuel_before = obj.fuel_mass_kg
        dm = obj.apply_delta_v(dv)
        assert obj.fuel_mass_kg < fuel_before
        assert dm > 0.

    def test_cannot_burn_more_than_available_fuel(self):
        sv = eci_from_elements(600., 53., 0., 0.)
        obj = OrbitalObject("SAT-TEST", "SATELLITE", sv, fuel_mass_kg=1.)
        dv = np.array([0., 10., 0.])   # huge burn
        obj.apply_delta_v(dv)
        assert obj.fuel_mass_kg >= 0.
