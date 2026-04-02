"""
AstraShield | core/data_gen.py
"Predict. Prevent. Protect."
══════════════════════════════════════════════════════════════════════
Synthetic Orbital Population Generator
50 satellites (Walker-delta) + 10,000 debris (6 fragmentation events
+ historical shells + background noise) with orbital DNA tagging.
══════════════════════════════════════════════════════════════════════
"""
import numpy as np
import pandas as pd
from core.physics import (OrbitalObject, StateVector,
                           eci_from_elements, circ_vel, RE)

np.random.seed(2026)

# Named fragmentation events — each debris piece tagged to its origin
FRAG_EVENTS = [
    # (name,          alt_km, inc°,  raan°, n_frags, spread_km, dv_kms)
    ("COSMOS-BREAKUP",   780, 72.0,  45.0,  1800, 55, 0.18),
    ("FENGYUN-SC01",     850, 98.6, 120.0,  1400, 70, 0.22),
    ("DELTA-K-REMNANT",  600, 51.6, 200.0,  1000, 40, 0.14),
    ("TELOS-COLLISION",  470, 65.0, 310.0,   700, 35, 0.12),
    ("AERO-CASCADE",     920, 74.0, 270.0,   600, 90, 0.25),
    ("MICROSAT-CHAIN",  1200, 82.0, 330.0,   500,110, 0.30),
]

SHELL_PARAMS = [
    (1500, 0, 360, 1000),
    ( 850,20, 160,  700),
    ( 300, 0, 360,  300),
]


def generate_satellites(n=50):
    planes, per_plane = 5, n // 5
    alt_shells = [550., 650., 780.]
    sats = []
    for p in range(planes):
        raan = p * (360/planes)
        alt  = alt_shells[p % 3]
        inc  = np.random.uniform(53, 98)
        for s in range(per_plane):
            nu  = s*(360/per_plane) + np.random.uniform(-1.5,1.5)
            sv  = eci_from_elements(alt, inc, raan, nu)
            fuel= np.random.uniform(38, 50)
            sats.append(OrbitalObject(
                obj_id=f"SAT-{p:02d}-{s:02d}", obj_type="SATELLITE",
                state=sv, fuel_mass_kg=fuel))
    return sats


def generate_debris(n=10000):
    dlist = []
    weights = np.array([e[4] for e in FRAG_EVENTS], dtype=float)
    weights /= weights.sum()
    n_frag  = int(n * 0.70)
    n_shell = int(n * 0.20)
    n_noise = n - n_frag - n_shell

    # ── Fragmentation clouds ──────────────────────────────────────
    for i,(name,alt,inc,raan,_,spread,dv) in enumerate(FRAG_EVENTS):
        count = int(n_frag * weights[i])
        for k in range(count):
            nu = np.random.uniform(0,360)
            sv = eci_from_elements(alt,inc,raan,nu)
            r  = sv.pos() + np.random.randn(3)*(spread/np.sqrt(3))
            v  = sv.vel() + np.random.randn(3)*dv
            dlist.append(OrbitalObject(
                obj_id=f"DEB-F{len(dlist):05d}", obj_type="DEBRIS",
                state=StateVector(*r,*v), fuel_mass_kg=0.,
                parent_event=name, generation=1,
                area_m2=np.random.uniform(0.01, 0.5)))   # debris vary in size

    # ── Historical shells ─────────────────────────────────────────
    for (alt,imin,imax,cnt) in SHELL_PARAMS:
        for _ in range(cnt):
            sv = eci_from_elements(
                alt+np.random.uniform(-80,80),
                np.random.uniform(imin,imax),
                np.random.uniform(0,360),
                np.random.uniform(0,360))
            r = sv.pos()+np.random.randn(3)*15
            v = sv.vel()+np.random.randn(3)*0.06
            dlist.append(OrbitalObject(
                obj_id=f"DEB-S{len(dlist):05d}", obj_type="DEBRIS",
                state=StateVector(*r,*v), fuel_mass_kg=0.,
                parent_event="HISTORICAL-SHELL",
                area_m2=np.random.uniform(0.001, 0.1)))

    # ── Background noise ──────────────────────────────────────────
    for _ in range(n_noise):
        sv = eci_from_elements(
            np.random.uniform(200,2000),
            np.random.uniform(0,98),
            np.random.uniform(0,360),
            np.random.uniform(0,360))
        v = sv.vel()+np.random.randn(3)*0.04
        dlist.append(OrbitalObject(
            obj_id=f"DEB-N{len(dlist):05d}", obj_type="DEBRIS",
            state=StateVector(*sv.pos(),*v), fuel_mass_kg=0.,
            parent_event="UNKNOWN",
            area_m2=np.random.uniform(0.0001, 0.05)))

    return dlist[:n]


def objects_to_dataframe(objs):
    rows = []
    for o in objs:
        s = o.state
        rows.append({
            "id": o.obj_id, "type": o.obj_type,
            "x":s.x,"y":s.y,"z":s.z,
            "vx":s.vx,"vy":s.vy,"vz":s.vz,
            "altitude_km": s.altitude(),
            "speed_kms":   s.speed(),
            "fuel_kg":     o.fuel_mass_kg,
            "area_m2":     o.area_m2,
            "parent_event":o.parent_event,
            "generation":  o.generation,
            "status":      o.status,
        })
    return pd.DataFrame(rows)
