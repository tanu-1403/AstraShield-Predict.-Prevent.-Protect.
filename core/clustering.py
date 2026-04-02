"""
AstraShield | core/clustering.py — HDBSCAN + BallTree O(N log N) CA
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import logging
logger = logging.getLogger("AstraShield.clustering")

try:
    from sklearn.cluster import HDBSCAN
    _HDB = True
except ImportError:
    _HDB = False

def _auto_eps(X,k=10,pct=85.):
    nbrs=NearestNeighbors(n_neighbors=k,algorithm="ball_tree").fit(X)
    d,_=nbrs.kneighbors(X); return float(np.percentile(d[:,-1],pct))

def run_clustering(debris_df,min_cluster_size=10):
    pos=debris_df[["x","y","z"]].values
    X=StandardScaler().fit_transform(pos)
    if _HDB:
        from sklearn.cluster import HDBSCAN
        model=HDBSCAN(min_cluster_size=min_cluster_size,min_samples=5,n_jobs=-1)
        labels=model.fit_predict(X); method="HDBSCAN"
    else:
        eps=_auto_eps(X,k=min_cluster_size)
        model=DBSCAN(eps=eps,min_samples=min_cluster_size,algorithm="ball_tree",n_jobs=-1)
        labels=model.fit_predict(X); method=f"DBSCAN(eps={eps:.4f})"
    out=debris_df.copy(); out["cluster_id"]=labels
    n=len(set(labels))-(1 if -1 in labels else 0)
    logger.info(f"[{method}] {n} clusters | {(labels==-1).sum():,} noise")
    return out,n,method

class ConjunctionAssessor:
    """BallTree O(N log N) conjunction screener."""
    def __init__(self,debris_df):
        self.debris_df=debris_df; self._tree=None; self._pos=None
    def build_index(self):
        self._pos=self.debris_df[["x","y","z"]].values
        self._tree=BallTree(self._pos,metric="euclidean",leaf_size=40)
        logger.info(f"[BallTree] Index built on {len(self._pos):,} debris")
    def screen_satellite(self,sat_pos,radius_km=100.):
        if self._tree is None: raise RuntimeError("Call build_index() first")
        idx=self._tree.query_radius(sat_pos.reshape(1,-1),r=radius_km)[0]
        if not len(idx): return pd.DataFrame()
        hits=self.debris_df.iloc[idx].copy()
        hits["miss_dist_km"]=np.linalg.norm(self._pos[idx]-sat_pos,axis=1)
        return hits.sort_values("miss_dist_km")
    def screen_all(self,sat_df,radius_km=100.):
        if self._tree is None: self.build_index()
        cdms=[]; sat_pos=sat_df[["x","y","z"]].values
        for i,(_,sat) in enumerate(sat_df.iterrows()):
            hits=self.screen_satellite(sat_pos[i],radius_km)
            if len(hits):
                hits["sat_id"]=sat["id"]; hits["sat_alt_km"]=sat["altitude_km"]
                cdms.append(hits)
        return pd.concat(cdms,ignore_index=True) if cdms else pd.DataFrame()

def _hull_vol(pos):
    if len(pos)<4: r=max(np.std(pos)*3,1.); return (4/3)*np.pi*r**3
    try: return max(ConvexHull(pos).volume,1.)
    except: r=np.max(np.linalg.norm(pos-pos.mean(0),axis=1)); return (4/3)*np.pi*max(r,1.)**3

def chan_pc(sigma_r,miss_dist,cr=0.01):
    s2=sigma_r**2
    if s2<1e-12: return 0.
    return (np.pi*cr**2/(2*np.pi*s2))*np.exp(-miss_dist**2/(2*s2))

def compute_cluster_stats(debris_df):
    clust=debris_df[debris_df["cluster_id"]>=0].copy()
    clust["speed"]=np.sqrt(clust.vx**2+clust.vy**2+clust.vz**2)
    rows=[]
    for cid,g in clust.groupby("cluster_id"):
        pos=g[["x","y","z"]].values; sig=np.std(np.linalg.norm(pos-pos.mean(0),axis=1))
        vol=_hull_vol(pos)
        parent=g["parent_event"].value_counts().idxmax() if "parent_event" in g.columns else "?"
        rows.append({"cluster_id":int(cid),"size":len(g),"volume_km3":vol,
            "density_proxy":len(g)/vol,"avg_speed_kms":g["speed"].mean(),
            "vel_disp_kms":g["speed"].std(),"sigma_r_km":sig,
            "avg_Pc":chan_pc(sig,2*sig),"mean_alt_km":g["altitude_km"].mean(),
            "dominant_parent":parent,
            "cx":pos[:,0].mean(),"cy":pos[:,1].mean(),"cz":pos[:,2].mean()})
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    def n01(s): r=s.max()-s.min(); return (s-s.min())/r if r>0 else s*0
    df["risk_score"]=0.50*n01(df["density_proxy"])+0.30*n01(df["vel_disp_kms"])+0.20*n01(df["avg_Pc"])
    df["risk_level"]=df["risk_score"].apply(lambda s:"HIGH" if s>=0.65 else "MEDIUM" if s>=0.30 else "LOW")
    return df.sort_values("risk_score",ascending=False).reset_index(drop=True)

def tag_risk(debris_df,stats_df):
    m=stats_df.set_index("cluster_id")[["risk_level","risk_score"]].to_dict("index")
    df=debris_df.copy()
    df["risk_level"]=df["cluster_id"].map(lambda c:m.get(c,{}).get("risk_level","NOISE"))
    df["risk_score"]=df["cluster_id"].map(lambda c:m.get(c,{}).get("risk_score",0.))
    return df
