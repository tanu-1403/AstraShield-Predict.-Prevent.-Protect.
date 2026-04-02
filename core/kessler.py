"""
AstraShield | core/kessler.py
"Predict. Prevent. Protect."
Kessler Cascade Monte Carlo Simulator — chain-reaction breakup propagation
"""
import numpy as np
import pandas as pd
from core.physics import RE

def frag_yield(mass_kg,vel_kms):
    """NASA Standard Breakup Model (simplified): N ~ 0.1 * M^0.75 * v^1.2"""
    return max(1,int(np.random.poisson(0.1*(mass_kg**0.75)*(vel_kms**1.2))))

def col_rate(density,v_kms=0.5,sigma_km=0.01):
    return density*np.pi*sigma_km**2*v_kms*86400

def run_cascade_mc(debris_df,stats_df,n_trials=300,max_gen=6,
                   runaway=5000,m_kg=200.,v_kms=10.):
    risk_cl=stats_df[stats_df["risk_level"].isin(["HIGH","MEDIUM"])]
    rows=[]
    for _,cl in risk_cl.iterrows():
        cid=int(cl["cluster_id"]); alt=cl["mean_alt_km"]; d0=cl["density_proxy"]
        totals=[]; runaways=[]; maxgens=[]
        for _ in range(n_trials):
            total=0; d=d0; runaway_hit=False; gen_counts=[]
            for gen in range(max_gen):
                rate=col_rate(d,v_kms); exp_col=rate
                if exp_col<0.01: break
                n_col=max(0,int(np.random.poisson(exp_col)))
                if n_col==0: break
                nf=sum(frag_yield(m_kg*(0.7**gen),v_kms) for _ in range(n_col))
                gen_counts.append(nf); total+=nf
                if total>runaway: runaway_hit=True; break
                d+=nf/(4*np.pi*(RE+alt)**2*100)
            totals.append(total); runaways.append(runaway_hit); maxgens.append(len(gen_counts))
        rows.append({
            "cluster_id":cid,"risk_level":cl["risk_level"],"mean_alt_km":alt,
            "P_runaway":np.mean(runaways),"mean_new_frags":np.mean(totals),
            "p95_new_frags":np.percentile(totals,95),
            "p95_generations":np.percentile(maxgens,95),
            "kessler_index":np.mean(runaways)*np.mean(totals)/runaway,
        })
    return pd.DataFrame(rows).sort_values("kessler_index",ascending=False)
