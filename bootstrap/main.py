"""AstraShield | main.py | "Predict. Prevent. Protect." """
import os,sys,time,logging
import numpy as np; import pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("data",exist_ok=True); os.makedirs("logs",exist_ok=True)
logging.basicConfig(level=logging.INFO,
    format="[AstraShield] %(asctime)s %(message)s",
    handlers=[logging.FileHandler("logs/astrashield.log"),logging.StreamHandler()])
logger=logging.getLogger("AstraShield")

C="\033[96m";G="\033[92m";Y="\033[93m";R="\033[91m"
W="\033[97m";D="\033[2m";RESET="\033[0m";BOLD="\033[1m"

def banner():
    print(f"\n{C}{'═'*72}{RESET}")
    print(f"{C}{BOLD}{'AstraShield':^72}{RESET}")
    print(f"{W}{'Predict. Prevent. Protect.':^72}{RESET}")
    print(f"{D}{'National Space Hackathon 2026  ·  IIT Delhi':^72}{RESET}")
    print(f"{C}{'═'*72}{RESET}\n")

def stage(n,tot,name):
    w=28; f=int((n/tot)*w)
    print(f"\n{C}{BOLD}[{n:02d}/{tot}]{RESET} {W}{name}{RESET}")
    print(f"  {C}[{'█'*f+'░'*(w-f)}] {n/tot:.0%}{RESET}")

def ok(m):  print(f"  {G}✓{RESET} {m}")
def info(m):print(f"  {D}→{RESET} {m}")
def warn(m):print(f"  {Y}⚡{RESET} {m}")

def run():
    t0=time.time(); banner(); TOT=10

    stage(1,TOT,"SYNTHETIC ORBITAL POPULATION GENERATION")
    from core.data_gen import generate_satellites,generate_debris,objects_to_dataframe
    sat_df=objects_to_dataframe(generate_satellites(50))
    deb_df_raw=objects_to_dataframe(generate_debris(10000))
    ok(f"50 satellites + 10,000 debris | speed {deb_df_raw.speed_kms.min():.2f}–{deb_df_raw.speed_kms.max():.2f} km/s")

    stage(2,TOT,"HDBSCAN ADAPTIVE CLUSTERING + RISK SCORING")
    from core.clustering import run_clustering,compute_cluster_stats,tag_risk
    deb_df,n_cl,method=run_clustering(deb_df_raw,min_cluster_size=10)
    stats_df=compute_cluster_stats(deb_df); deb_df=tag_risk(deb_df,stats_df)
    rc=deb_df["risk_level"].value_counts().to_dict()
    ok(f"{n_cl} clusters via {method} | HIGH={rc.get('HIGH',0):,} MED={rc.get('MEDIUM',0):,} LOW={rc.get('LOW',0):,}")

    stage(3,TOT,"BALLTREE O(N log N) CONJUNCTION ASSESSMENT")
    from core.clustering import ConjunctionAssessor
    ca=ConjunctionAssessor(deb_df); ca.build_index()
    cdm_df=ca.screen_all(sat_df,radius_km=200.)
    ok(f"BallTree on {len(deb_df):,} objects | {len(cdm_df):,} conjunctions within 200 km")
    if len(cdm_df)>0: info(f"Closest: {cdm_df['miss_dist_km'].min():.2f} km")

    stage(4,TOT,"KESSLER CASCADE MONTE CARLO (300 trials/cluster)")
    from core.kessler import run_cascade_mc
    kessler_df=run_cascade_mc(deb_df,stats_df,n_trials=300,max_gen=6)
    if len(kessler_df)>0:
        ok(f"{len(kessler_df)} clusters analysed | {(kessler_df['P_runaway']>0.5).sum()} with P(runaway)>50%")
    else:
        warn("No HIGH/MEDIUM clusters for cascade analysis")
        kessler_df=pd.DataFrame(columns=["cluster_id","risk_level","mean_alt_km","P_runaway","mean_new_frags","p95_new_frags","p95_generations","kessler_index"])

    stage(5,TOT,"CMA-ES SELF-TUNING MANEUVER OPTIMIZER (80 iterations)")
    from core.cmaes_optimizer import optimize_maneuver
    from core.physics import StateVector,rtn_to_eci,eci_from_elements
    cmaes_hist=[]; best_miss=0.; best_dv=0.
    sat_sv_orig=sat_sv_ev=deb_sv=None
    hi=stats_df[stats_df["risk_level"]=="HIGH"]
    if len(hi)>0 and len(sat_df)>0:
        cl=hi.iloc[0]; cp=np.array([cl.cx,cl.cy,cl.cz])
        sp=sat_df[["x","y","z"]].values
        sr=sat_df.iloc[np.argmin(np.linalg.norm(sp-cp,axis=1))]
        sat_sv_orig=StateVector(sr.x,sr.y,sr.z,sr.vx,sr.vy,sr.vz)
        sd=deb_df[deb_df["cluster_id"]==int(cl.cluster_id)]
        deb_sv=StateVector(cl.cx,cl.cy,cl.cz,sd.vx.mean(),sd.vy.mean(),sd.vz.mean())
        bc,cmaes_hist,sr2=optimize_maneuver(sat_sv_orig,deb_sv,tca_s=3600.,max_iter=80,popsize=20)
        best_miss=sr2["miss_dist_km"]; best_dv=sr2["total_dv_kms"]
        M=rtn_to_eci(sat_sv_orig); dv=M@bc[1:4]
        sat_sv_ev=StateVector(sat_sv_orig.x,sat_sv_orig.y,sat_sv_orig.z,*(sat_sv_orig.vel()+dv))
        ok(f"CMA-ES converged | Miss: {best_miss:.2f} km | ΔV: {best_dv*1000:.1f} m/s")
    else:
        warn("No HIGH-risk cluster — demo orbit used")
        sat_sv_orig=eci_from_elements(600,53,45,0)
        deb_sv=eci_from_elements(600,53.1,45.05,0.5)
        sat_sv_ev=sat_sv_orig
        cmaes_hist=list(np.cumsum(np.random.uniform(0.01,0.05,80)))

    stage(6,TOT,"GHOST ORBIT PREDICTION (T+24h)")
    from core.triage import predict_ghost_positions
    ghost_df=predict_ghost_positions(deb_df,horizon_s=86400.,sample_n=2000)
    ok(f"{len(ghost_df):,} ghost positions computed")
    if len(ghost_df)>0: info(f"Max drift: {ghost_df['drift_km'].max():.1f} km")

    stage(7,TOT,"COLLISION PROBABILITY HEAT ATLAS")
    from core.triage import build_heat_atlas
    alt_e,inc_e,heat=build_heat_atlas(deb_df)
    ok(f"Atlas: {heat.shape} grid | {(~np.isnan(heat)).sum()} cells | peak log₁₀Pc={np.nanmax(heat):.2f}")

    stage(8,TOT,"FUEL TRIAGE ENGINE")
    from core.triage import compute_triage
    triage_df=compute_triage(sat_df,deb_df,stats_df)
    ok(f"MANEUVER={( triage_df.action=='MANEUVER').sum()} GRAVEYARD={(triage_df.action=='GRAVEYARD').sum()} ABANDON={(triage_df.action=='ABANDON').sum()}")

    stage(9,TOT,"EXPORTING DATASETS")
    deb_df.to_csv("data/debris_clustered.csv",index=False)
    sat_df.to_csv("data/satellites.csv",index=False)
    stats_df.to_csv("data/cluster_risk_stats.csv",index=False)
    kessler_df.to_csv("data/kessler_results.csv",index=False)
    triage_df.to_csv("data/triage_results.csv",index=False)
    ghost_df.to_csv("data/ghost_orbits.csv",index=False)
    if len(cdm_df)>0: cdm_df.to_csv("data/conjunction_events.csv",index=False)
    ok("All CSVs saved to data/")

    stage(10,TOT,"GENERATING PUBLICATION FIGURES")
    from viz.visualizer import plot_dashboard,plot_kessler,plot_cmaes,plot_atlas
    info("Fig 1: Mission Control Dashboard...")
    plot_dashboard(deb_df,sat_df,stats_df,"data/fig1_astrashield_dashboard.png")
    if len(kessler_df)>0:
        info("Fig 2: Kessler Cascade...")
        plot_kessler(kessler_df,"data/fig2_astrashield_kessler.png")
    if sat_sv_orig and deb_sv:
        info("Fig 3: CMA-ES Maneuver...")
        plot_cmaes(cmaes_hist,best_miss,best_dv,sat_sv_orig,sat_sv_ev,deb_sv,"data/fig3_astrashield_cmaes.png")
    info("Fig 4: Heat Atlas + Ghost + Triage...")
    plot_atlas(alt_e,inc_e,heat,ghost_df,triage_df,"data/fig4_astrashield_atlas.png")
    ok("All figures saved")

    print()
    from viz.terminal import print_dashboard
    print_dashboard(deb_df,sat_df,stats_df,kessler_df,triage_df)

    elapsed=time.time()-t0
    print(f"\n{C}{'═'*72}{RESET}")
    print(f"{G}{BOLD}  AstraShield pipeline complete in {elapsed:.1f}s{RESET}")
    print(f"{C}{'═'*72}{RESET}\n")
    for f in sorted(os.listdir("data")):
        sz=os.path.getsize(f"data/{f}")
        print(f"  {C}→{RESET} data/{f:<45} {D}{sz/1024:.1f} KB{RESET}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

if __name__=="__main__":
    run()
