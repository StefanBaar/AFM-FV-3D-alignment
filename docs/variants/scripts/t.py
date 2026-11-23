import sys, json, numpy as np
sys.path.insert(0,'/home/claude/out'); sys.path.insert(0,'/home/claude/docs_build/ting')
import afm_ting_pipeline as AP, common as C
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import savgol_filter
P=dict(C.P, radius=2.5)
def prep(i):
    a=C.raw(i); t,s,F,zz,sig=AP.prep(a,P,C.B)
    ip=int(np.argmax(F)); Fs=savgol_filter(F,9,2); ic=np.where(Fs[:ip]<max(4*sig,0.03*Fs[ip]))[0][-1]
    ie=ip+np.where(Fs[ip:]<0)[0][0]; return t,s,F,sig,ic,ip,ie
def fitA(t,s,F,ic,ie,a0=0.3,dz=0.0,win=None):
    win=win if win is not None else ie
    f=lambda p: AP.ting_model(p,t,s,F,win,P)
    return least_squares(f,[a0,s[ic]+dz],bounds=([0,s[ic]-4],[0.95,s[ic]+1.5]),diff_step=[1e-2,1e-5],x_scale=[.1,.1])
idx=list(range(7,2775,93))[:30]
out=[]
for i in idx:
    try: t,s,F,sig,ic,ip,ie=prep(i)
    except Exception: continue
    r={}
    # T1 multistart
    sols=[]
    for a0 in (0.05,0.2,0.4,0.6,0.8,0.95):
        for dz in (0,-0.4):
            q=fitA(t,s,F,ic,ie,a0,dz); sols.append((q.cost,q.x[0],q.x[1]))
    sols.sort(); r['best']=sols[0]; r['spread_alpha_within_1pct_cost']=float(np.ptp([x[1] for x in sols if x[0]<=sols[0][0]*1.01]))
    r['default']=fitA(t,s,F,ic,ie).x.tolist(); r['default_cost']=float(fitA(t,s,F,ic,ie).cost)
    # profile cost(alpha) minimising over zc
    A=np.linspace(0.05,0.95,19); prof=[]
    for a in A:
        g=lambda zc: 0.5*np.sum(AP.ting_model([a,zc],t,s,F,ie,P)**2)
        m=minimize_scalar(g,bounds=(s[ic]-3,s[ic]+1.5),method='bounded',options={'xatol':1e-4}); prof.append(m.fun)
    prof=np.array(prof); r['prof']=prof.tolist()
    # residual noise for a chi2 scale; approx CI = alpha range with dcost < 0.5*sig_res^2*... use dchi2<1 with chi2=2cost/s2
    s2=2*sols[0][0]/ie; ok=A[2*(prof-prof.min())/s2<1.0]; r['ci']=[float(ok.min()),float(ok.max())] if len(ok) else [np.nan,np.nan]
    # T3 approach only vs full
    q=fitA(t,s,F,ic,ip+1); r['alpha_app_only']=float(q.x[0])
    out.append(r); print(i, 'best α %.3f default α %.3f  multistart spread %.3f  CI %s  app-only %.3f'%(r['best'][1],r['default'][0],r['spread_alpha_within_1pct_cost'],np.round(r['ci'],2),r['alpha_app_only']),flush=True)
json.dump(out,open('t.json','w'))
