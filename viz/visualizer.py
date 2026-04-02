"""
AstraShield | viz/visualizer.py
"Predict. Prevent. Protect."
4-figure publication-quality mission dashboard.
Aesthetic: Deep-space tactical — void black, phosphor cyan, amber alerts.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

BG="#020C18"; BG2="#040F1E"; GRID="#071626"
CYAN="#00E5FF"; AMBER="#FFB300"; RED="#FF2D55"
GREEN="#00FF9C"; FROST="#C8DDEE"; DIM="#2A4A6A"
BRAND="#00E5FF"

RISK_C={"HIGH":RED,"MEDIUM":AMBER,"LOW":GREEN,"NOISE":DIM}

LOGO = [
    " █████╗ ███████╗████████╗██████╗  █████╗ ",
    "██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗",
    "███████║███████╗   ██║   ██████╔╝███████║",
    "██╔══██║╚════██║   ██║   ██╔══██╗██╔══██║",
    "██║  ██║███████║   ██║   ██║  ██║██║  ██║",
    "╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝",
    "  S H I E L D  ·  Predict. Prevent. Protect.",
]

def _ax(ax,title="",xlabel="",ylabel=""):
    ax.set_facecolor(BG2)
    ax.tick_params(colors=FROST,labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor(DIM); sp.set_linewidth(0.5)
    ax.grid(color=GRID,linewidth=0.4,linestyle="--",alpha=0.9)
    if title: ax.set_title(title,color=CYAN,fontsize=9,fontweight="bold",
                           fontfamily="monospace",pad=7)
    if xlabel: ax.set_xlabel(xlabel,color=FROST,fontsize=8,fontfamily="monospace")
    if ylabel: ax.set_ylabel(ylabel,color=FROST,fontsize=8,fontfamily="monospace")

def _earth(ax,alpha=0.35):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0,0),6378.137,color="#071E38",alpha=alpha,zorder=0))

def plot_dashboard(debris_df,sat_df,stats_df,save="fig1_astrashield_dashboard.png"):
    fig=plt.figure(figsize=(24,20),facecolor=BG)
    gs=gridspec.GridSpec(3,3,figure=fig,hspace=0.38,wspace=0.28,
                         top=0.91,bottom=0.04,left=0.05,right=0.97)
    ax3d=fig.add_subplot(gs[0,:2],projection="3d")
    ax_h =fig.add_subplot(gs[0,2])
    ax_xy=fig.add_subplot(gs[1,0])
    ax_rs=fig.add_subplot(gs[1,1])
    ax_vd=fig.add_subplot(gs[1,2])
    ax_dna=fig.add_subplot(gs[2,0])
    ax_fg=fig.add_subplot(gs[2,1])
    ax_ca=fig.add_subplot(gs[2,2])

    for ax in [ax_h,ax_xy,ax_rs,ax_vd,ax_dna,ax_fg,ax_ca]: _ax(ax)
    ax3d.set_facecolor(BG)
    for sp in [ax3d.xaxis,ax3d.yaxis,ax3d.zaxis]:
        sp.pane.fill=False; sp.pane.set_edgecolor(DIM)
    ax3d.tick_params(colors=DIM,labelsize=6)

    # ── Logo watermark top-left ───────────────────────────────────
    for i,line in enumerate(LOGO):
        fig.text(0.01,0.98-i*0.012,line,color=CYAN,fontsize=5.5,
                 fontfamily="monospace",alpha=0.8)

    # ── 3D ECI ───────────────────────────────────────────────────
    R=6378.137; u,v=np.mgrid[0:2*np.pi:18j,0:np.pi:10j]
    ax3d.plot_wireframe(R*np.cos(u)*np.sin(v),R*np.sin(u)*np.sin(v),
                        R*np.cos(v),color="#0A2035",lw=0.25,alpha=0.5)
    for lv in ["NOISE","LOW","MEDIUM","HIGH"]:
        sub=debris_df[debris_df["risk_level"]==lv]
        if not len(sub): continue
        s={"HIGH":1.5,"MEDIUM":1.0,"LOW":0.5,"NOISE":0.15}[lv]
        ax3d.scatter(sub.x,sub.y,sub.z,c=RISK_C[lv],s=s,alpha=0.4,rasterized=True)
    ax3d.scatter(sat_df.x,sat_df.y,sat_df.z,c=CYAN,s=22,marker="^",
                 zorder=10,edgecolors="#FFFFFF44",linewidths=0.3)
    ax3d.set_title("ECI ORBITAL FIELD — RISK STRATIFICATION",
                   color=CYAN,fontsize=10,fontfamily="monospace",pad=5)
    patches=[mpatches.Patch(color=c,label=l) for l,c in RISK_C.items()]
    patches.append(mpatches.Patch(color=CYAN,label="Satellites"))
    ax3d.legend(handles=patches,fontsize=7,facecolor=BG2,
                labelcolor=FROST,loc="upper left",markerscale=1.5,framealpha=0.8)

    # ── Altitude histogram ────────────────────────────────────────
    bins=np.linspace(200,2000,45)
    for lv in ["LOW","MEDIUM","HIGH","NOISE"]:
        sub=debris_df[debris_df["risk_level"]==lv]
        ax_h.hist(sub["altitude_km"],bins=bins,color=RISK_C[lv],
                  alpha=0.75,label=lv,edgecolor="none")
    for alt in [550,650,780]:
        ax_h.axvline(alt,color=CYAN,lw=0.7,ls="--",alpha=0.6)
    _ax(ax_h,"ALTITUDE DISTRIBUTION BY RISK","Altitude (km)","Fragments")
    ax_h.legend(facecolor=BG2,labelcolor=FROST,fontsize=8)

    # ── XY cross-section ─────────────────────────────────────────
    _earth(ax_xy)
    for lv in ["NOISE","LOW","MEDIUM","HIGH"]:
        sub=debris_df[debris_df["risk_level"]==lv]
        s={"HIGH":1.8,"MEDIUM":1.0,"LOW":0.5,"NOISE":0.12}[lv]
        ax_xy.scatter(sub.x,sub.y,c=RISK_C[lv],s=s,alpha=0.3,rasterized=True)
    ax_xy.scatter(sat_df.x,sat_df.y,c=CYAN,s=16,marker="^",zorder=5)
    ax_xy.set_aspect("equal")
    _ax(ax_xy,"EQUATORIAL PLANE (X–Y)","X (km)","Y (km)")

    # ── Risk bubble ───────────────────────────────────────────────
    sc=ax_rs.scatter(stats_df["size"],stats_df["risk_score"],
        s=np.clip(stats_df["density_proxy"]*8e6,20,800),
        c=stats_df["risk_score"],cmap="RdYlGn_r",vmin=0,vmax=1,
        alpha=0.88,edgecolors="#FFFFFF22",linewidths=0.4)
    for _,r in stats_df.iterrows():
        ax_rs.annotate(f'C{int(r["cluster_id"])}',
            (r["size"],r["risk_score"]),color=FROST,fontsize=7,
            ha="center",va="center",fontfamily="monospace")
    cb=plt.colorbar(sc,ax=ax_rs,pad=0.01)
    cb.ax.tick_params(colors=FROST,labelsize=7)
    cb.set_label("Risk Score",color=FROST,fontsize=8)
    ax_rs.axhspan(0.65,1.,alpha=0.06,color=RED)
    ax_rs.axhspan(0.30,0.65,alpha=0.06,color=AMBER)
    ax_rs.axhspan(0.,0.30,alpha=0.06,color=GREEN)
    _ax(ax_rs,"CLUSTER RISK SCORE vs SIZE","Cluster Size","Risk Score [0–1]")

    # ── Velocity dispersion ───────────────────────────────────────
    cc=[RISK_C.get(l,DIM) for l in stats_df["risk_level"]]
    ax_vd.scatter(stats_df["mean_alt_km"],stats_df["vel_disp_kms"],
        c=cc,s=np.clip(stats_df["size"]/8+15,10,200),
        edgecolors="#FFFFFF22",linewidths=0.4,alpha=0.9)
    patches2=[mpatches.Patch(color=c,label=l) for l,c in RISK_C.items() if l!="NOISE"]
    ax_vd.legend(handles=patches2,facecolor=BG2,labelcolor=FROST,fontsize=8)
    _ax(ax_vd,"VEL DISPERSION vs ALTITUDE","Mean Alt (km)","Speed Std (km/s)")

    # ── Debris DNA pie ────────────────────────────────────────────
    if "parent_event" in debris_df.columns:
        vc=debris_df["parent_event"].value_counts().head(7)
        colors=[CYAN,AMBER,RED,GREEN,FROST,"#FF6B9D","#9B59B6"]
        wedges,texts,auts=ax_dna.pie(
            vc.values,labels=None,colors=colors[:len(vc)],
            autopct="%1.1f%%",pctdistance=0.8,startangle=90,
            textprops={"color":FROST,"fontsize":7,"fontfamily":"monospace"})
        ax_dna.legend(wedges,[t[:18] for t in vc.index],
            loc="lower center",fontsize=6.5,facecolor=BG2,
            labelcolor=FROST,framealpha=0.9,ncol=2,bbox_to_anchor=(0.5,-0.25))
        _ax(ax_dna,"DEBRIS DNA — FRAGMENTATION LINEAGE")
        ax_dna.grid(False)

    # ── Fuel gauge ────────────────────────────────────────────────
    if "fuel_kg" in sat_df.columns:
        fv=sat_df["fuel_kg"].values
        fc=[GREEN if f>30 else AMBER if f>15 else RED for f in fv]
        ax_fg.barh(range(len(fv)),fv,color=fc,edgecolor="none",height=0.7)
        ax_fg.axvline(50,color=DIM,lw=0.5,ls="--")
        ax_fg.axvline(15,color=AMBER,lw=0.8,ls="--",alpha=0.7)
        ax_fg.axvline(5,color=RED,lw=0.8,ls="--",alpha=0.7)
        ax_fg.set_yticks(range(len(fv)))
        ax_fg.set_yticklabels([f"S{i:02d}" for i in range(len(fv))],
            fontsize=5,color=FROST,fontfamily="monospace")
        _ax(ax_fg,"FLEET FUEL RESERVES","Propellant (kg)","Satellite")

    # ── Conjunction assessment bar ────────────────────────────────
    rc=debris_df["risk_level"].value_counts()
    lvls=["HIGH","MEDIUM","LOW","NOISE"]
    vals=[rc.get(l,0) for l in lvls]
    cols=[RISK_C[l] for l in lvls]
    ax_ca.barh(lvls,vals,color=cols,edgecolor="none",height=0.6)
    for i,(l,v) in enumerate(zip(lvls,vals)):
        ax_ca.text(v+10,i,f"{v:,}",color=FROST,va="center",fontsize=8,
                   fontfamily="monospace")
    _ax(ax_ca,"DEBRIS RISK CLASSIFICATION","Fragment Count","Risk Level")

    # ── Branding ──────────────────────────────────────────────────
    fig.suptitle(
        "◈  AstraShield  ·  Predict. Prevent. Protect.  ·  Mission Control Dashboard",
        color=CYAN,fontsize=13,fontweight="bold",fontfamily="monospace",y=0.97)
    fig.text(0.5,0.005,
        "National Space Hackathon 2026  ·  IIT Delhi  ·  Orbital Debris Intelligence System",
        ha="center",color=DIM,fontsize=8,fontfamily="monospace")

    plt.savefig(save,dpi=150,bbox_inches="tight",facecolor=BG)
    print(f"[AstraShield] Saved → {save}")
    return fig

def plot_kessler(kessler_df,save="fig2_astrashield_kessler.png"):
    fig,axes=plt.subplots(1,3,figsize=(20,7),facecolor=BG)
    fig.subplots_adjust(wspace=0.32,left=0.07,right=0.96,top=0.87,bottom=0.12)
    ax1,ax2,ax3=axes
    for ax in axes: _ax(ax)

    xr=range(len(kessler_df))
    bar_cols=[RED if p>0.5 else AMBER if p>0.2 else GREEN for p in kessler_df["P_runaway"]]
    ax1.bar(xr,kessler_df["P_runaway"],color=bar_cols,edgecolor="none",width=0.7)
    ax1.set_xticks(xr)
    ax1.set_xticklabels([f"C{int(c)}" for c in kessler_df["cluster_id"]],
        fontsize=8,color=FROST,fontfamily="monospace")
    ax1.axhline(0.5,color=RED,lw=1,ls="--",alpha=0.7)
    ax1.set_ylim(0,1.05)
    _ax(ax1,"P(KESSLER RUNAWAY) BY CLUSTER","Cluster","Probability")

    sc=ax2.scatter(kessler_df["kessler_index"],kessler_df["mean_new_frags"],
        s=kessler_df["p95_new_frags"]/20+30,c=kessler_df["P_runaway"],
        cmap="hot",vmin=0,vmax=1,edgecolors="#FFFFFF33",linewidths=0.5)
    cb=plt.colorbar(sc,ax=ax2,pad=0.01)
    cb.set_label("P(Runaway)",color=FROST,fontsize=8); cb.ax.tick_params(colors=FROST)
    for _,r in kessler_df.iterrows():
        ax2.annotate(f'C{int(r["cluster_id"])}',
            (r["kessler_index"],r["mean_new_frags"]),color=FROST,fontsize=7,ha="center")
    _ax(ax2,"NEW FRAGMENTS vs KESSLER INDEX","Kessler Index","Mean New Fragments")

    ax3.scatter(kessler_df["mean_alt_km"],kessler_df["p95_generations"],
        c=bar_cols,s=120,edgecolors=FROST,linewidths=0.5,zorder=5)
    for _,r in kessler_df.iterrows():
        ax3.annotate(f'C{int(r["cluster_id"])}\n{r["P_runaway"]:.0%}',
            (r["mean_alt_km"],r["p95_generations"]),
            color=FROST,fontsize=6.5,ha="center",va="bottom",fontfamily="monospace")
    _ax(ax3,"CASCADE DEPTH vs ALTITUDE","Mean Altitude (km)","Max Generations (p95)")

    fig.suptitle("◈  AstraShield  ·  Kessler Cascade Monte Carlo Analysis",
        color=RED,fontsize=13,fontweight="bold",fontfamily="monospace")
    plt.savefig(save,dpi=150,bbox_inches="tight",facecolor=BG)
    print(f"[AstraShield] Saved → {save}")

def plot_cmaes(history,miss_km,dv_kms,sat_sv,sat_ev,deb_sv,
               save="fig3_astrashield_cmaes.png"):
    from core.physics import propagate
    fig,axes=plt.subplots(1,2,figsize=(17,8),facecolor=BG)
    fig.subplots_adjust(wspace=0.3,left=0.08,right=0.96,top=0.87,bottom=0.10)
    ax_g,ax_o=axes
    for ax in axes: _ax(ax)

    gens=np.arange(len(history))
    ax_g.plot(gens,history,color=CYAN,lw=2,zorder=5)
    ax_g.fill_between(gens,min(history),history,color=CYAN,alpha=0.08)
    ax_g.axhline(max(history),color=AMBER,lw=0.8,ls="--",alpha=0.7)
    ax_g.text(len(history)*0.97,max(history),f"Peak: {max(history):.3f}",
        color=AMBER,fontsize=8,ha="right",va="bottom",fontfamily="monospace")
    info=f"Miss Distance : {miss_km:.2f} km\nΔV Total      : {dv_kms*1000:.1f} m/s\nAlgorithm     : CMA-ES"
    ax_g.text(0.05,0.95,info,transform=ax_g.transAxes,color=FROST,fontsize=9,
        va="top",fontfamily="monospace",
        bbox=dict(boxstyle="round",fc=BG2,ec=DIM,lw=0.6,alpha=0.9))
    _ax(ax_g,"CMA-ES MANEUVER OPTIMIZER — CONVERGENCE",
        "Generation","Best Fitness Score")

    _earth(ax_o)
    def trace(sv,col,lbl,ls="-"):
        pts=[sv.pos()[:2]]
        cur=sv
        for _ in range(60):
            cur=propagate(cur,90,3)
            pts.append(cur.pos()[:2])
        pts=np.array(pts)
        ax_o.plot(pts[:,0],pts[:,1],color=col,lw=1.3,ls=ls,label=lbl)
    trace(sat_sv,GREEN,"Original Orbit","--")
    trace(sat_ev,CYAN,"Evasion Orbit")
    trace(deb_sv,RED,"Debris Track")
    ax_o.scatter(*sat_sv.pos()[:2],c=GREEN,s=55,zorder=10,marker="o")
    ax_o.scatter(*sat_ev.pos()[:2],c=CYAN,s=55,zorder=10,marker="^")
    ax_o.scatter(*deb_sv.pos()[:2],c=RED,s=55,zorder=10,marker="x")
    ax_o.legend(facecolor=BG2,labelcolor=FROST,fontsize=8,loc="upper right")
    ax_o.set_aspect("equal")
    _ax(ax_o,"MANEUVER GEOMETRY — ORBITAL PLANE","X (km)","Y (km)")

    fig.suptitle("◈  AstraShield  ·  CMA-ES Optimal Evasion Maneuver",
        color=CYAN,fontsize=13,fontweight="bold",fontfamily="monospace")
    plt.savefig(save,dpi=150,bbox_inches="tight",facecolor=BG)
    print(f"[AstraShield] Saved → {save}")

def plot_atlas(alt_e,inc_e,heat,ghost_df,triage_df,
               save="fig4_astrashield_atlas.png"):
    fig=plt.figure(figsize=(24,9),facecolor=BG)
    gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.28,
                         left=0.05,right=0.97,top=0.87,bottom=0.10)
    ax_h=fig.add_subplot(gs[0,0]); ax_g=fig.add_subplot(gs[0,1])
    ax_t=fig.add_subplot(gs[0,2])
    for ax in [ax_h,ax_g,ax_t]: _ax(ax)

    masked=np.ma.masked_invalid(heat)
    im=ax_h.pcolormesh(inc_e,alt_e,masked,cmap="inferno",vmin=-12,vmax=-3,shading="auto")
    cb=plt.colorbar(im,ax=ax_h,pad=0.01)
    cb.set_label("log₁₀(Pc/day)",color=FROST,fontsize=8); cb.ax.tick_params(colors=FROST,labelsize=7)
    ax_h.axhline(780,color=GREEN,lw=0.8,ls="--",alpha=0.8)
    ax_h.axhline(850,color=AMBER,lw=0.8,ls="--",alpha=0.8)
    ax_h.text(3,788,"Cosmos belt",color=GREEN,fontsize=7,fontfamily="monospace")
    ax_h.text(3,858,"Fengyun belt",color=AMBER,fontsize=7,fontfamily="monospace")
    _ax(ax_h,"COLLISION PROBABILITY HEAT ATLAS","Inclination (deg)","Altitude (km)")

    if len(ghost_df)>0:
        for lv in ["NOISE","LOW","MEDIUM","HIGH"]:
            sub=ghost_df[ghost_df["risk_level"]==lv]
            s={"HIGH":4,"MEDIUM":2.5,"LOW":1.5,"NOISE":0.5}[lv]
            ax_g.scatter(sub["ghost_lon"],sub["ghost_lat"],
                c=RISK_C[lv],s=s,alpha=0.45,rasterized=True)
    ax_g.axhline(0,color=DIM,lw=0.4,alpha=0.5); ax_g.axvline(0,color=DIM,lw=0.4,alpha=0.5)
    ax_g.set_xlim(-180,180); ax_g.set_ylim(-90,90)
    _ax(ax_g,"GHOST ORBIT POSITIONS (T+24h)","Longitude (deg)","Latitude (deg)")

    if len(triage_df)>0:
        top=triage_df.head(25)
        ac={a:c for a,c in [("MANEUVER",GREEN),("GRAVEYARD",AMBER),("ABANDON",RED)]}
        bc=[ac.get(a,DIM) for a in top["action"]]
        sc=top["triage_score"]/top["triage_score"].max()
        ax_t.barh(range(len(top)),sc,color=bc,edgecolor="none",height=0.7)
        ax_t.set_yticks(range(len(top)))
        ax_t.set_yticklabels(top["sat_id"].values,fontsize=7,color=FROST,
            fontfamily="monospace")
        patches=[mpatches.Patch(color=c,label=a) for a,c in ac.items()]
        ax_t.legend(handles=patches,facecolor=BG2,labelcolor=FROST,
            fontsize=8,loc="lower right")
        _ax(ax_t,"FUEL TRIAGE — SATELLITE PRIORITY","Normalized Score","Satellite")

    fig.suptitle(
        "◈  AstraShield  ·  Ghost Orbits  ·  Heat Atlas  ·  Fuel Triage Engine",
        color=AMBER,fontsize=13,fontweight="bold",fontfamily="monospace")
    plt.savefig(save,dpi=150,bbox_inches="tight",facecolor=BG)
    print(f"[AstraShield] Saved → {save}")
