import sys, numpy as np
sys.path.insert(0,'/home/claude/out'); sys.path.insert(0,'/home/claude/docs_build/ting')
import afm_ting_pipeline as AP, common as C
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
P=dict(C.P, radius=2.5); rng=np.random.default_rng(1)
a=C.raw(103); t,s,F,zz,sig=AP.prep(a,P,C.B)
ip0=int(np.argmax(F)); Fs0=savgol_filter(F,9,2); ic0=np.where(Fs0[:ip0]<max(4*sig,0.03*Fs0[ip0]))[0][-1]
zc=s[ic0]+0.7
v=np.gradient(savgol_filter(s,15,2),t)
def fit(Fsyn,approach_only):
    ip=int(np.argmax(Fsyn)); Fs=savgol_filter(Fsyn,9,2); ic=np.where(Fs[:ip]<max(4*sig,0.03*Fs[ip]))[0][-1]
    neg=np.where(Fs[ip:]<0)[0]; ie=ip+neg[0] if len(neg) else len(Fsyn)
    win=ip+1 if approach_only else ie; best=None
    for dz in (0,-0.4):
        r=least_squares(lambda p:AP.ting_model(p,t,s,Fsyn,win,P),[0.3,s[ic]+dz],bounds=([0,s[ic]-4],[0.95,s[ic]+1.5]),diff_step=[1e-2,1e-5],x_scale=[.1,.1])
        if best is None or r.cost<best.cost: best=r
    return best.x[0]
for al in (0.2,0.35,0.5):
    g=AP.ting_model([al,zc],t,s,np.ones_like(s),len(s),P,True)
    E1=4.0/max(g[1].max(),1e-12)                       # scale to 4 nN peak
    Fc=g[1]/max(g[0],1e-30)*E1 if False else None
    gg=np.zeros_like(s); 
    # pure Ting shape scaled
    d=np.maximum(s-zc,0); i0=np.where(s[:int(np.argmax(s))+1]<=zc)[0][-1]; seg=d[i0:].copy(); seg[0]=0
    gg[i0:]=AP.ting_shape(seg,al,t[1]-t[0],P); Fting=gg*4.0/gg.max()
    hyd=4.7e-4*v/(np.maximum(zc-s,0)+0.27)            # A from the squeeze-film fits (eta ~4 mPa s, R 2.5 um), h0 0.27 um
    for lab,Fsyn in (('Ting only',Fting),('Ting + toe (squeeze film)',Fting+hyd)):
        Fn=Fsyn+sig*rng.standard_normal(len(s))
        print(f'true alpha {al:.2f}  {lab:26s} fitted: full {fit(Fn,False):.3f}   approach only {fit(Fn,True):.3f}')
print('--- squeeze-film variant on the same synthetic curves (with toe) ---')
for al in (0.2,0.35,0.5):
    gg=np.zeros_like(s); d=np.maximum(s-zc,0); i0=np.where(s[:int(np.argmax(s))+1]<=zc)[0][-1]; seg=d[i0:].copy(); seg[0]=0
    gg[i0:]=AP.ting_shape(seg,al,t[1]-t[0],P); Fsyn=gg*4.0/gg.max()+4.7e-4*v/(np.maximum(zc-s,0)+0.27)+sig*rng.standard_normal(len(s))
    ip=int(np.argmax(Fsyn)); Fs=savgol_filter(Fsyn,9,2); ic=np.where(Fs[:ip]<max(4*sig,0.03*Fs[ip]))[0][-1]; ie=ip+np.where(Fs[ip:]<0)[0][0]
    main=least_squares(lambda p:AP.ting_model(p,t,s,Fsyn,ie,P),[0.3,s[ic]],bounds=([0,s[ic]-4],[0.95,s[ic]+1.5]),diff_step=[1e-2,1e-5],x_scale=[.1,.1]).x
    vr=AP.fit_variants(t,s,Fsyn,s,sig,1.0,P,ic,ip,ie,main,('squeeze','labcp'))
    print(f"true {al:.2f}: main {main[0]:.3f} | squeeze {vr['alpha_sq']:.3f} (CP error {vr['zc_sq']-zc:+.2f} um, h0 {vr['h0_sq']:.2f}) | lab CP {vr['alpha_labcp']:.3f}")
