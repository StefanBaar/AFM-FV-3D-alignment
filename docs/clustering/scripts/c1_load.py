"""Load the 41-frame time-lapse maps into one pixel table (features + geometry + cell info)."""
import numpy as np, json, glob, os
D='/home/claude/seg2/maps'; PX,PY=1.0135,0.5
names=sorted(os.path.basename(f)[:-4] for f in glob.glob(D+'/E/*.npy'))
C=json.load(open('/home/claude/stage/curv.json'))
def tmin(t):h,m,s=map(int,t.split(':'));return h*60+m+s/60
T0=tmin(C[0]['time'])
rows=[]
for k,n in enumerate(names):
    L=lambda key:np.load(f'{D}/{key}/{n}.npy').astype(float)
    E1,a,dl,tp=L('E'),L('alpha'),L('delta'),L('topo'); ex=np.load(f'{D}/excluded/{n}.npy')
    ny,nx=E1.shape; iy,ix=np.mgrid[0:ny,0:nx]
    gy,gx=np.gradient(tp,PY,PX); slope=np.degrees(np.arctan(np.hypot(gx,gy)))
    G=np.array(C[k]['G']); P=np.stack([ix*PX,iy*PY,tp,np.ones_like(tp)],-1)@G.T
    rec=dict(frame=np.full(E1.shape,k),t_h=np.full(E1.shape,(tmin(C[k]['time'])-T0)/60),ix=ix,iy=iy,x=ix*PX,y=iy*PY,
        X=P[...,0],Y=P[...,1],Z=P[...,2],topo=tp,slope=slope,excl=ex.astype(float),
        logE=np.log10(E1*1000**a),alpha=a,logd=np.log10(dl),
        Fadh=L('adh_force_nN'),logW=np.log10(np.clip(L('adh_work_fJ'),1e-3,None)),Ladh=L('adh_length_um'),
        pull=L('adh_pulloff_um'),nrup=L('adh_n_ruptures'),teth=(L('adh_tether_nN')>0).astype(float),
        tethF=L('adh_tether_nN'),tethL=L('adh_tether_len_um'),detach=L('adh_detached'),
        cell=L('cells'),bnd=L('cell_boundary'))
    rows.append({kk:v.ravel() for kk,v in rec.items()})
tab={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
tab['ok']=(tab['excl']==0)&np.isfinite(tab['logE'])&np.isfinite(tab['alpha'])&np.isfinite(tab['logd'])&np.isfinite(tab['Fadh'])&np.isfinite(tab['logW'])&np.isfinite(tab['Ladh'])
np.savez_compressed('/home/claude/docs_build/clustering/pix.npz',names=np.array(names),**tab)
print(len(tab['ok']),'pixels,',tab['ok'].sum(),'valid')
