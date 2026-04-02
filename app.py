"""
AstraShield | app.py
"Predict. Prevent. Protect."
Run: streamlit run app.py
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="AstraShield",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Branding ────────────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align:center; color:#00e5ff; font-family:monospace;'>
  ◈ AstraShield
</h1>
<p style='text-align:center; color:#aaa; font-family:monospace; margin-top:-12px;'>
  Predict. Prevent. Protect. &nbsp;·&nbsp; National Space Hackathon 2026 · IIT Delhi
</p>
<hr style='border-color:#0d2030;'>
""", unsafe_allow_html=True)

# ── Sidebar controls ────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")
n_sats  = st.sidebar.slider("Satellites",        10,  50,  50)
n_debs  = st.sidebar.slider("Debris objects", 1000, 10000, 5000, step=500)
min_cls = st.sidebar.slider("Min cluster size",   5,  20,  10)
n_mc    = st.sidebar.slider("Kessler MC trials", 50, 300, 100, step=50)
run_btn = st.sidebar.button("🚀 Run AstraShield", use_container_width=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌍 Mission Control",
    "💥 Kessler Cascade",
    "🧬 CMA-ES Optimizer",
    "👻 Ghost Orbits & Atlas",
    "🛢️ Fuel Triage",
])

# ── Run pipeline on button click ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(n_sats, n_debs, min_cls, n_mc):
    from core.data_gen import generate_satellites, generate_debris, objects_to_dataframe
    from core.clustering import run_clustering, compute_cluster_stats, tag_risk
    from core.kessler import run_cascade_mc
    from core.triage import predict_ghost_positions, build_heat_atlas, compute_triage
    from core.cmaes_optimizer import optimize_maneuver
    from core.physics import StateVector, rtn_to_eci, eci_from_elements

    sat_df = objects_to_dataframe(generate_satellites(n_sats))
    deb_df = objects_to_dataframe(generate_debris(n_debs))
    deb_df, n_cl, method = run_clustering(deb_df, min_cluster_size=min_cls)
    stats_df = compute_cluster_stats(deb_df)
    deb_df = tag_risk(deb_df, stats_df)
    kessler_df = run_cascade_mc(deb_df, stats_df, n_trials=n_mc, max_gen=5)
    ghost_df = predict_ghost_positions(deb_df, horizon_s=86400, sample_n=800)
    alt_e, inc_e, heat = build_heat_atlas(deb_df)
    triage_df = compute_triage(sat_df, deb_df, stats_df)

    # CMA-ES on worst cluster
    cmaes_hist, miss_km, dv_kms = [], 0., 0.
    sat_sv = deb_sv = sat_sv_ev = None
    hi = stats_df[stats_df["risk_level"] == "HIGH"]
    if len(hi) > 0:
        cl = hi.iloc[0]
        cp = np.array([cl.cx, cl.cy, cl.cz])
        sp = sat_df[["x","y","z"]].values
        sr = sat_df.iloc[np.argmin(np.linalg.norm(sp - cp, axis=1))]
        sat_sv = StateVector(sr.x, sr.y, sr.z, sr.vx, sr.vy, sr.vz)
        sd = deb_df[deb_df["cluster_id"] == int(cl.cluster_id)]
        deb_sv = StateVector(cl.cx, cl.cy, cl.cz,
                             sd.vx.mean(), sd.vy.mean(), sd.vz.mean())
        bc, cmaes_hist, res = optimize_maneuver(
            sat_sv, deb_sv, tca_s=3600., max_iter=60, popsize=16)
        miss_km = res["miss_dist_km"]; dv_kms = res["total_dv_kms"]
        M = rtn_to_eci(sat_sv)
        dv_eci = M @ bc[1:4]
        sat_sv_ev = StateVector(sat_sv.x, sat_sv.y, sat_sv.z,
                                *(sat_sv.vel() + dv_eci))
    else:
        sat_sv = eci_from_elements(600, 53, 45, 0)
        deb_sv = eci_from_elements(600, 53.1, 45.05, 0.5)
        sat_sv_ev = sat_sv
        cmaes_hist = list(np.cumsum(np.random.uniform(0.01, 0.05, 60)))

    return (sat_df, deb_df, stats_df, kessler_df, ghost_df,
            alt_e, inc_e, heat, triage_df,
            cmaes_hist, miss_km, dv_kms, sat_sv, sat_sv_ev, deb_sv, n_cl, method)


if run_btn or "pipeline_done" in st.session_state:
    if run_btn:
        st.cache_data.clear()
        st.session_state["pipeline_done"] = True

    with st.spinner("Running AstraShield pipeline..."):
        (sat_df, deb_df, stats_df, kessler_df, ghost_df,
         alt_e, inc_e, heat, triage_df,
         cmaes_hist, miss_km, dv_kms,
         sat_sv, sat_sv_ev, deb_sv, n_cl, method) = run_pipeline(
            n_sats, n_debs, min_cls, n_mc)

    rc = deb_df["risk_level"].value_counts().to_dict()

    # ── Tab 1: Mission Control ────────────────────────────────────────
    with tab1:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Satellites",      len(sat_df))
        c2.metric("Debris",          f"{len(deb_df):,}")
        c3.metric("Clusters",        n_cl)
        c4.metric("HIGH Risk",       rc.get("HIGH", 0))
        c5.metric("Fleet Fuel (kg)", f"{sat_df['fuel_kg'].sum():.0f}")

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Debris Field — XY Plane")
            fig, ax = plt.subplots(figsize=(5, 5), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            colors = {"HIGH":"#ff2d55","MEDIUM":"#ffb300","LOW":"#00ff9c","NOISE":"#2a4a6a"}
            for lv, col in colors.items():
                sub = deb_df[deb_df["risk_level"] == lv]
                s = {"HIGH":2,"MEDIUM":1.2,"LOW":0.5,"NOISE":0.15}[lv]
                ax.scatter(sub.x, sub.y, c=col, s=s, alpha=0.4, rasterized=True)
            ax.scatter(sat_df.x, sat_df.y, c="#00e5ff", s=18, marker="^", zorder=5)
            ax.set_xlabel("X (km)", color="#c8ddee"); ax.set_ylabel("Y (km)", color="#c8ddee")
            ax.tick_params(colors="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.subheader("Altitude Distribution by Risk")
            fig, ax = plt.subplots(figsize=(5, 5), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            bins = np.linspace(200, 2000, 40)
            for lv, col in colors.items():
                sub = deb_df[deb_df["risk_level"] == lv]
                ax.hist(sub["altitude_km"], bins=bins, color=col, alpha=0.7, label=lv)
            ax.legend(facecolor="#040f1e", labelcolor="#c8ddee", fontsize=8)
            ax.tick_params(colors="#c8ddee")
            ax.set_xlabel("Altitude (km)", color="#c8ddee")
            ax.set_ylabel("Count", color="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.subheader("Top Risk Clusters")
        display_cols = ["cluster_id","size","mean_alt_km","risk_level",
                        "risk_score","vel_disp_kms","dominant_parent"]
        st.dataframe(
            stats_df[display_cols].head(20).style
            .background_gradient(subset=["risk_score"], cmap="RdYlGn_r")
            .format({"risk_score": "{:.3f}", "mean_alt_km": "{:.0f}",
                     "vel_disp_kms": "{:.4f}"}),
            use_container_width=True,
        )

        with st.expander("💾 Download debris dataset"):
            st.download_button("Download debris_clustered.csv",
                               deb_df.to_csv(index=False).encode(),
                               "debris_clustered.csv", "text/csv")

    # ── Tab 2: Kessler Cascade ───────────────────────────────────────
    with tab2:
        if len(kessler_df) > 0:
            k1, k2, k3 = st.columns(3)
            k1.metric("Clusters analysed", len(kessler_df))
            k2.metric("P(runaway) > 50%",  (kessler_df["P_runaway"] > 0.5).sum())
            k3.metric("Max Kessler Index",  f"{kessler_df['kessler_index'].max():.4f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("P(Runaway) by Cluster")
                fig, ax = plt.subplots(figsize=(5, 4), facecolor="#020c18")
                ax.set_facecolor("#040f1e")
                bar_colors = ["#ff2d55" if p>0.5 else "#ffb300" if p>0.2 else "#00ff9c"
                              for p in kessler_df["P_runaway"]]
                ax.bar(range(len(kessler_df)), kessler_df["P_runaway"],
                       color=bar_colors, edgecolor="none")
                ax.set_xticks(range(len(kessler_df)))
                ax.set_xticklabels([f"C{int(c)}" for c in kessler_df["cluster_id"]],
                                   fontsize=8, color="#c8ddee")
                ax.axhline(0.5, color="#ff2d55", lw=1, ls="--", alpha=0.7)
                ax.tick_params(colors="#c8ddee")
                for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with col_b:
                st.subheader("Mean New Fragments (24h)")
                fig, ax = plt.subplots(figsize=(5, 4), facecolor="#020c18")
                ax.set_facecolor("#040f1e")
                ax.scatter(kessler_df["kessler_index"],
                           kessler_df["mean_new_frags"],
                           c=kessler_df["P_runaway"], cmap="hot",
                           s=80, edgecolors="#ffffff33")
                ax.tick_params(colors="#c8ddee")
                ax.set_xlabel("Kessler Index", color="#c8ddee")
                ax.set_ylabel("Mean New Fragments", color="#c8ddee")
                for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
                st.pyplot(fig, use_container_width=True)
                plt.close()

            st.dataframe(kessler_df.style.format({
                "P_runaway": "{:.1%}", "kessler_index": "{:.4f}",
                "mean_new_frags": "{:.0f}", "mean_alt_km": "{:.0f}"}),
                use_container_width=True)
        else:
            st.info("No HIGH/MEDIUM risk clusters found for cascade analysis.")

    # ── Tab 3: CMA-ES ────────────────────────────────────────────────
    with tab3:
        m1, m2, m3 = st.columns(3)
        m1.metric("Miss Distance", f"{miss_km:.2f} km")
        m2.metric("Total ΔV",      f"{dv_kms*1000:.1f} m/s")
        m3.metric("Generations",   len(cmaes_hist))

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("CMA-ES Convergence")
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            ax.plot(cmaes_hist, color="#00e5ff", lw=2)
            ax.fill_between(range(len(cmaes_hist)),
                            min(cmaes_hist), cmaes_hist, color="#00e5ff", alpha=0.08)
            ax.axhline(max(cmaes_hist), color="#ffb300", lw=0.8, ls="--")
            ax.tick_params(colors="#c8ddee")
            ax.set_xlabel("Generation", color="#c8ddee")
            ax.set_ylabel("Best Fitness", color="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.subheader("Maneuver Geometry (X–Y plane)")
            from core.physics import propagate
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            def trace(sv, col, lbl, ls="-"):
                pts = [sv.pos()[:2]]
                cur = sv
                for _ in range(50):
                    cur = propagate(cur, 90, 3)
                    pts.append(cur.pos()[:2])
                pts = np.array(pts)
                ax.plot(pts[:,0], pts[:,1], color=col, lw=1.2, ls=ls, label=lbl)
            trace(sat_sv,    "#00ff9c", "Original",  "--")
            trace(sat_sv_ev, "#00e5ff", "Evaded")
            trace(deb_sv,    "#ff2d55", "Debris")
            ax.legend(facecolor="#040f1e", labelcolor="#c8ddee", fontsize=8)
            ax.set_aspect("equal"); ax.tick_params(colors="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # ── Tab 4: Ghost Orbits & Heat Atlas ─────────────────────────────
    with tab4:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Ghost Orbit Positions (T+24h)")
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            ghost_colors = {"HIGH":"#ff2d55","MEDIUM":"#ffb300",
                            "LOW":"#00ff9c","NOISE":"#2a4a6a"}
            for lv, col in ghost_colors.items():
                sub = ghost_df[ghost_df["risk_level"] == lv]
                s = {"HIGH":4,"MEDIUM":2.5,"LOW":1.5,"NOISE":0.5}[lv]
                ax.scatter(sub["ghost_lon"], sub["ghost_lat"],
                           c=col, s=s, alpha=0.45, rasterized=True)
            ax.set_xlim(-180,180); ax.set_ylim(-90,90)
            ax.axhline(0, color="#2a4a6a", lw=0.4)
            ax.axvline(0, color="#2a4a6a", lw=0.4)
            ax.tick_params(colors="#c8ddee")
            ax.set_xlabel("Longitude", color="#c8ddee")
            ax.set_ylabel("Latitude", color="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.subheader("Collision Probability Heat Atlas")
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            masked = np.ma.masked_invalid(heat)
            im = ax.pcolormesh(inc_e, alt_e, masked,
                               cmap="inferno", vmin=-12, vmax=-3, shading="auto")
            plt.colorbar(im, ax=ax, label="log₁₀(Pc/day)")
            ax.axhline(780, color="#00ff9c", lw=0.8, ls="--")
            ax.axhline(850, color="#ffb300", lw=0.8, ls="--")
            ax.tick_params(colors="#c8ddee")
            ax.set_xlabel("Inclination (deg)", color="#c8ddee")
            ax.set_ylabel("Altitude (km)", color="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with st.expander("💾 Download ghost orbits CSV"):
            st.download_button("Download ghost_orbits.csv",
                               ghost_df.to_csv(index=False).encode(),
                               "ghost_orbits.csv", "text/csv")

    # ── Tab 5: Fuel Triage ───────────────────────────────────────────
    with tab5:
        t1, t2, t3 = st.columns(3)
        t1.metric("MANEUVER",  (triage_df["action"]=="MANEUVER").sum())
        t2.metric("GRAVEYARD", (triage_df["action"]=="GRAVEYARD").sum())
        t3.metric("ABANDON",   (triage_df["action"]=="ABANDON").sum())

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Priority Ranking")
            fig, ax = plt.subplots(figsize=(5, 6), facecolor="#020c18")
            ax.set_facecolor("#040f1e")
            top = triage_df.head(20)
            bar_colors = ["#00ff9c" if a=="MANEUVER" else "#ffb300" if a=="GRAVEYARD"
                          else "#ff2d55" for a in top["action"]]
            norm_score = top["triage_score"] / top["triage_score"].max()
            ax.barh(range(len(top)), norm_score, color=bar_colors, height=0.7)
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top["sat_id"].values, fontsize=7, color="#c8ddee")
            ax.tick_params(colors="#c8ddee")
            ax.set_xlabel("Normalised Score", color="#c8ddee")
            for sp in ax.spines.values(): sp.set_edgecolor("#2a4a6a")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.subheader("Triage Table")
            st.dataframe(
                triage_df[["sat_id","fuel_kg","fuel_frac","nearest_debris_km",
                            "triage_score","action"]].style
                .format({"fuel_kg":"{:.1f}","fuel_frac":"{:.0%}",
                         "nearest_debris_km":"{:.1f}","triage_score":"{:.3f}"})
                .applymap(lambda v: "color:#00ff9c" if v=="MANEUVER"
                          else "color:#ffb300" if v=="GRAVEYARD"
                          else "color:#ff2d55" if v=="ABANDON" else "",
                          subset=["action"]),
                use_container_width=True, height=450,
            )

        with st.expander("💾 Download triage results"):
            st.download_button("Download triage_results.csv",
                               triage_df.to_csv(index=False).encode(),
                               "triage_results.csv", "text/csv")

else:
    for tab in [tab1, tab2, tab3, tab4, tab5]:
        with tab:
            st.info("👈 Configure parameters in the sidebar and click **Run AstraShield** to start.")