"""Controls: (1) cell identity after removing tissue-scale trends; (2) model-variant dependence of the clustering (single full map)."""
import numpy as np, json
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
d=dict(np.load('../pix.npz')); z=np.load('../labels.npz'); X=z['X']; m=z['m']; fr=d['frame']; nf=int(fr.max())+1; feats=list(z['feats'])
PX,PY=1.0135,0.5; out={}
for sig in [3.0,6.0]:
    Xr=np.full_like(X,np.nan)
    for k in range(nf):
        s=np.where(fr==k)[0]; ny=int(d['iy'][s].max())+1; nx=int(d['ix'][s].max())+1; M=m[s].reshape(ny,nx)
        for i in range(X.shape[1]):
            G=np.where(M,X[s,i].reshape(ny,nx),0.); w=gaussian_filter(M.astype(float),(sig/PY,sig/PX)); sm=gaussian_filter(G,(sig/PY,sig/PX))/np.maximum(w,1e-6)
            Xr[s,i]=np.where(M,G-sm,np.nan).ravel()
    rows=[]
    for k in range(nf):
        s=m&(fr==k)&(d['cell']>0)
        for t in np.unique(d['cell'][s]).astype(int):
            q=s&(d['cell']==t)
            if q.sum()>=4: rows.append([k,t]+[np.median(Xr[q,i]) for i in range(len(feats))])
    R=np.array(rows); tr=R[:,1].astype(int); u,c=np.unique(tr,return_counts=True); keep=np.isin(tr,u[c>=4]); res={}
    for i,f in enumerate(feats):
        vv,tt=R[keep,2+i],tr[keep]; g=[vv[tt==a] for a in np.unique(tt)]; n0=np.mean([len(x) for x in g])
        MSB=sum(len(x)*(x.mean()-vv.mean())**2 for x in g)/(len(g)-1); MSW=sum(((x-x.mean())**2).sum() for x in g)/(len(vv)-len(g)); res[f]=float((MSB-MSW)/(MSB+(n0-1)*MSW))
    out[f'ICC_detrended_sigma{sig:g}um']=res
# shuffled-track null: random pairing of cell observations within frames keeps ICC ~0
# (2) variants on the single full map
F=np.load('/home/claude/fullv/home_claude_data.npz',allow_pickle=True)
ok=np.isfinite(F['E_at_1ms_Pa'])&np.isfinite(F['adh_force_nN'])&np.isfinite(F['alpha'])&(F['R2']>0.9)&np.isfinite(F['E_at_1ms_sq_Pa'])&np.isfinite(F['E_at_1ms_labcp_Pa'])
def z_(v): v=v[ok]; return (v-np.median(v))/(1.4826*np.median(np.abs(v-np.median(v))))
adh=np.column_stack([z_(F['adh_force_nN']),z_(np.log10(np.clip(F['adh_work_fJ'],1e-3,None))),z_(F['adh_length_um']),z_((F['adh_tether_nN']>0).astype(float)+1e-9*np.random.default_rng(0).standard_normal(F['adh_force_nN'].shape))])
labs={}
for v,(ek,ak) in {'main':('E_at_1ms_Pa','alpha'),'squeeze':('E_at_1ms_sq_Pa','alpha_sq'),'lab CP':('E_at_1ms_labcp_Pa','alpha_labcp'),'vertical':('E_at_1ms_vertical_Pa','alpha_vertical')}.items():
    Xv=np.column_stack([z_(np.log10(F[ek])),z_(F[ak]),adh[:,0],adh[:,2]])
    labs[v]=GaussianMixture(4,covariance_type='full',n_init=5,random_state=0).fit_predict(Xv)
    Xe=Xv[:,:2]; labs[v+'_Ea']=GaussianMixture(3,covariance_type='full',n_init=5,random_state=0).fit_predict(Xe)
out['variant_ARI_vs_main']={v:float(ARI(labs['main'],labs[v])) for v in ['squeeze','lab CP','vertical']}
out['variant_ARI_vs_main_Ealpha_only']={v:float(ARI(labs['main_Ea'],labs[v+'_Ea'])) for v in ['squeeze','lab CP','vertical']}
out['variant_n']=int(ok.sum())
json.dump(out,open('../facts_controls.json','w'),indent=1); print(json.dumps(out,indent=1))
