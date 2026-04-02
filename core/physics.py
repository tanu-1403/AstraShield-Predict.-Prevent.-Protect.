"""
AstraShield | core/physics.py
"Predict. Prevent. Protect."
J2 + Atmospheric Drag RK4 Propagator — ECI frame, km/km/s
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

RE = 6378.137; MU = 398600.4418; J2 = 1.08263e-3; G0 = 9.80665e-3
OMEGA_E = 7.2921150e-5  # Earth rotation, rad/s

_ATM = np.array([
    [200,2.789e-10,6.369],[300,1.916e-11,7.293],[400,2.803e-12,8.770],
    [500,5.215e-13,10.792],[600,1.137e-13,13.243],[700,3.070e-14,16.457],
    [800,1.136e-14,21.524],[900,5.759e-15,27.559],[1000,3.561e-15,29.740],
    [1200,1.474e-15,45.546],[1500,5.194e-16,62.196],[2000,1.199e-16,90.628],
])

def atm_density(alt_km):
    if alt_km > 2000: return 0.0
    alt_km = max(alt_km, 200)
    idx = int(np.searchsorted(_ATM[:,0], alt_km)) - 1
    idx = np.clip(idx, 0, len(_ATM)-2)
    return _ATM[idx,1] * np.exp(-(alt_km - _ATM[idx,0]) / _ATM[idx,2])

@dataclass
class StateVector:
    x:float; y:float; z:float; vx:float; vy:float; vz:float
    def pos(self): return np.array([self.x,self.y,self.z])
    def vel(self): return np.array([self.vx,self.vy,self.vz])
    def as_array(self): return np.array([self.x,self.y,self.z,self.vx,self.vy,self.vz])
    def altitude(self): return np.linalg.norm(self.pos()) - RE
    def speed(self): return np.linalg.norm(self.vel())
    def to_dict(self): return {"x":self.x,"y":self.y,"z":self.z,"vx":self.vx,"vy":self.vy,"vz":self.vz}
    @staticmethod
    def from_array(a): return StateVector(*a)

@dataclass
class OrbitalObject:
    obj_id:str; obj_type:str; state:StateVector
    dry_mass_kg:float=500.0; fuel_mass_kg:float=50.0; isp_s:float=300.0
    cd:float=2.2; area_m2:float=4.0; status:str="NOMINAL"
    parent_event:Optional[str]=None; generation:int=0
    @property
    def total_mass(self): return self.dry_mass_kg + self.fuel_mass_kg
    @property
    def fuel_fraction(self): return self.fuel_mass_kg / 50.0
    @property
    def bstar(self): return self.cd * (self.area_m2*1e-6) / (2 * self.total_mass*1e-3)
    def apply_delta_v(self, dv_eci):
        dv_mag = np.linalg.norm(dv_eci)
        dm = min(self.total_mass*(1-np.exp(-dv_mag/(self.isp_s*G0))), self.fuel_mass_kg)
        self.fuel_mass_kg -= dm
        v = self.state.vel() + dv_eci
        self.state.vx,self.state.vy,self.state.vz = v
        return dm

def _a_j2(r):
    x,y,z = r; rm = np.linalg.norm(r); z2=(z/rm)**2
    a2b = -MU/rm**3*r
    f = 1.5*J2*MU*RE**2/rm**5
    return a2b + f*np.array([x*(5*z2-1),y*(5*z2-1),z*(5*z2-3)])

def _a_drag(r,v,bstar):
    alt = np.linalg.norm(r)-RE; rho = atm_density(alt)
    if rho==0: return np.zeros(3)
    vr = v - np.cross([0,0,OMEGA_E],r); vm = np.linalg.norm(vr)
    return -bstar*rho*vm*vr if vm>1e-9 else np.zeros(3)

def eom(s, bstar=0.0):
    r,v=s[:3],s[3:]; a=_a_j2(r)
    if bstar>0: a+=_a_drag(r,v,bstar)
    return np.concatenate([v,a])

def rk4_step(s,dt,bstar=0.0):
    k1=eom(s,bstar); k2=eom(s+.5*dt*k1,bstar)
    k3=eom(s+.5*dt*k2,bstar); k4=eom(s+dt*k3,bstar)
    return s+(dt/6)*(k1+2*k2+2*k3+k4)

def propagate(sv,dt_s,substeps=10,bstar=0.0):
    s=sv.as_array(); sub=dt_s/max(substeps,1)
    for _ in range(substeps): s=rk4_step(s,sub,bstar)
    return StateVector.from_array(s)

def propagate_trajectory(sv,total_s,step_s=60.,bstar=0.0):
    states,cur,el=[sv],sv,0.
    while el<total_s:
        dt=min(step_s,total_s-el); cur=propagate(cur,dt,max(4,int(dt/15)),bstar)
        states.append(cur); el+=dt
    return states

def rtn_to_eci(sv):
    r,v=sv.pos(),sv.vel(); R=r/np.linalg.norm(r)
    N=np.cross(r,v); N/=np.linalg.norm(N); T=np.cross(N,R)
    return np.column_stack([R,T,N])

def eci_to_geodetic(r):
    rm=np.linalg.norm(r)
    lat=np.degrees(np.arcsin(np.clip(r[2]/rm,-1,1)))
    lon=((np.degrees(np.arctan2(r[1],r[0]))+180)%360)-180
    return lat,lon,rm-RE

def circ_vel(alt): return np.sqrt(MU/(RE+alt))

def eci_from_elements(alt,inc_d,raan_d,nu_d):
    rm=RE+alt; vm=circ_vel(alt)
    inc,raan,nu=np.radians(inc_d),np.radians(raan_d),np.radians(nu_d)
    rp=rm*np.array([np.cos(nu),np.sin(nu),0.])
    vp=vm*np.array([-np.sin(nu),np.cos(nu),0.])
    cr,sr,ci,si=np.cos(raan),np.sin(raan),np.cos(inc),np.sin(inc)
    R=np.array([[cr,-sr*ci,sr*si],[sr,cr*ci,-cr*si],[0.,si,ci]])
    return StateVector(*R@rp,*R@vp)

def tca(sv1,sv2,dt_s=86400.,coarse=60.,fine=1.):
    """Two-pass TCA finder: coarse scan then fine refinement."""
    best_t,best_d=0.,np.inf
    s1,s2,t=sv1,sv2,0.
    while t<=dt_s:
        d=np.linalg.norm(s1.pos()-s2.pos())
        if d<best_d: best_d,best_t=d,t
        s1=propagate(s1,coarse,4); s2=propagate(s2,coarse,4); t+=coarse
    t0=max(0,best_t-2*coarse); t1=min(dt_s,best_t+2*coarse)
    s1=propagate(sv1,t0,max(4,int(t0/30))); s2=propagate(sv2,t0,max(4,int(t0/30)))
    t=t0
    while t<=t1:
        d=np.linalg.norm(s1.pos()-s2.pos())
        if d<best_d: best_d,best_t=d,t
        s1=propagate(s1,fine,2); s2=propagate(s2,fine,2); t+=fine
    return best_t,best_d
