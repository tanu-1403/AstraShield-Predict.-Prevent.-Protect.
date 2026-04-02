"""
AstraShield | core/triage.py
"Predict. Prevent. Protect."
Ghost Orbit Prediction + Collision Heat Atlas + Fuel Triage Engine
"""
import numpy as np
import pandas as pd
from core.physics import propagate, StateVector, eci_to_geodetic, RE

def predict_ghost_positions(debris_df,horizon_s=86400.,sample_n=2000):
    """Propagate debris T+24h. Returns ghost lat/lon/alt DataFrame."""
    sample=debris_df.sample(min(sample_n,len(debris_df)),random_state=42)
    rows=[]
    for _,row in sample.iterrows():
        sv=StateVector(row.x,row.y,row.z,row.vx,row.vy,row.vz)
        bstar=row.get("bstar",0.) if hasattr(row,"get") else 0.
        try:
            sf=propagate(sv,horizon_s,substeps=max(6,int(horizon_s/300)),bstar=bstar)
            lat,lon,alt=eci_to_geodetic(sf.pos())
            rows.append({"id":row.id,"cluster_id":row.get("cluster_id",-1),
                "risk_level":row.get("risk_level","NOISE"),
                "ghost_x":sf.x,"ghost_y":sf.y,"ghost_z":sf.z,
                "ghost_lat":lat,"ghost_lon":lon,"ghost_alt_km":alt,
                "orig_alt_km":row.altitude_km,
                "drift_km":np.linalg.norm(sf.pos()-sv.pos())})
        except Exception:
            pass
    return pd.DataFrame(rows)

def build_heat_atlas(debris_df,alt_bins=30,inc_bins=24):
    """Altitude x Inclination collision probability heat map."""
    v=np.sqrt(debris_df.vx**2+debris_df.vy**2+debris_df.vz**2)
    inc=np.degrees(np.arcsin(np.clip(np.abs(debris_df.vz)/np.clip(v,0.01,None),0,1)))
    alt_e=np.linspace(200,2000,alt_bins+1); inc_e=np.linspace(0,105,inc_bins+1)
    H=np.zeros((alt_bins,inc_bins))
    for i,(a0,a1) in enumerate(zip(alt_e[:-1],alt_e[1:])):
        for j,(n0,n1) in enumerate(zip(inc_e[:-1],inc_e[1:])):
            m=((debris_df.altitude_km>=a0)&(debris_df.altitude_km<a1)&(inc>=n0)&(inc<n1))
            nf=m.sum()
            if nf<2: H[i,j]=np.nan; continue
            r_mid=RE+(a0+a1)/2; dr=a1-a0; di=(n1-n0)/180
            vol=4*np.pi*r_mid**2*dr*di
            dens=nf/max(vol,1.)
            Pc=min(np.pi*0.01**2*dens*0.5*86400,1.)
            H[i,j]=np.log10(max(Pc,1e-12))
    return alt_e,inc_e,H

def compute_triage(sat_df,debris_df,stats_df):
    """
    Multi-criteria satellite priority queue.
    Score = (threat × mission_value × fuel_viability) / est_dv_cost
    Action: MANEUVER | GRAVEYARD | ABANDON
    """
    from sklearn.neighbors import NearestNeighbors
    sat_pos=sat_df[["x","y","z"]].values
    deb_pos=debris_df[["x","y","z"]].values
    nn=NearestNeighbors(n_neighbors=1,algorithm="ball_tree").fit(deb_pos)
    dists,_=nn.kneighbors(sat_pos); min_dist=dists[:,0]

    rows=[]
    for i,(_,row) in enumerate(sat_df.iterrows()):
        dist=min_dist[i] if i<len(min_dist) else 9999.
        alt=row.get("altitude_km",600.); fk=row.get("fuel_kg",50.)
        ff=fk/50.
        threat=1./(dist+0.1)
        mval=max(0.1,1.-(alt-200)/1800)
        cost=max(0.001,0.015*np.exp(-dist/10.))
        score=(threat*mval*ff)/cost
        rows.append({
            "sat_id":row.get("id",f"SAT-{i}"),
            "altitude_km":alt,"fuel_kg":fk,"fuel_frac":ff,
            "nearest_debris_km":dist,"triage_score":score,
            "action":"MANEUVER" if ff>0.10 else "GRAVEYARD" if ff>0.03 else "ABANDON"
        })
    df=pd.DataFrame(rows).sort_values("triage_score",ascending=False)
    df["priority_rank"]=range(1,len(df)+1)
    return df
