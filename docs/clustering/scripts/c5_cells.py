"""Cell-scale structure: feature profiles from junction to centre, and cell-level identity over time (ICC)."""
import numpy as np, json, csv, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
d=dict(np.load('../pix.npz')); z=np.load('../labels.npz'); X=z['X']; m=z['m']; L=z['L']; fr=d['frame']; nf=int(fr.max())+1
feats=list(z['feats']); names={'logE':'log E','alpha':'α','dstar':'δ*','adh1':'adh. strength','adh2':'adh. short vs long'}
PX,PY=1.0135,0.5
dist=np.full(len(fr),np.nan)
for k in range(nf):
    s=np.where(fr==k)[0]; ny=int(d['iy'][s].max())+1; nx=int(d['ix'][s].max())+1; C=d['cell'][s].reshape(ny,nx)
    dt=distance_transform_edt(C!=0,sampling=(PY,PX)); dt[C<0]=np.nan; dist[s]=dt.ravel()
bins=np.array([-.01,.01,.6,1.2,1.8,2.6]); lab=['junction','0–0.6','0.6–1.2','1.2–1.8','>1.8']
prof={}; ok=m&np.isfinite(dist)
for i,f in enumerate(feats):
    v=X[:,i]; prof[f]=[]
    for b in range(len(bins)-1):
        s=ok&(dist>bins[b])&(dist<=bins[b+1]); prof[f].append((float(np.nanmedian(v[s])),float(1.2533*np.nanstd(v[s])/np.sqrt(s.sum())),int(s.sum())))
# cluster composition vs distance
K=int(L.max())+1; comp=[[float(np.mean(L[ok&(dist>bins[b])&(dist<=bins[b+1])]==j)) for j in range(K)] for b in range(len(bins)-1)]
fig,ax=plt.subplots(1,2,figsize=(8.6,3))
for i,f in enumerate(feats):
    y=[p[0] for p in prof[f]]; e=[p[1] for p in prof[f]]; ax[0].errorbar(range(len(lab)),y,e,marker='o',ms=3,label=names[f]+(' (used for segmentation)' if f.startswith('adh') else ''))
ax[0].axhline(0,c='k',lw=.5);ax[0].set_xticks(range(len(lab)),lab,fontsize=7.5);ax[0].set_xlabel('distance from nearest junction (µm)');ax[0].set_ylabel('median (robust z)');ax[0].legend(frameon=False,fontsize=6.5)
cols=plt.get_cmap('tab10').colors; bot=np.zeros(len(lab)); NAMES=json.load(open('../cluster_names.json'))
for j in range(K): v=np.array([c[j] for c in comp]); ax[1].bar(range(len(lab)),v,bottom=bot,color=cols[j],label=f'C{j}');bot+=v
ax[1].set_xticks(range(len(lab)),lab,fontsize=7.5);ax[1].set_ylabel('fraction');ax[1].set_xlabel('distance from nearest junction (µm)');ax[1].legend(frameon=False,fontsize=7,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.15))
plt.tight_layout();plt.savefig('../fig/junction_profiles.pdf');plt.close()
# ---- cell-level: mean feature per (frame, track); ICC = between-track variance share (frame effects removed)
rows=[]
for k in range(nf):
    s=m&(fr==k)&(d['cell']>0)
    for t in np.unique(d['cell'][s]).astype(int):
        q=s&(d['cell']==t)
        if q.sum()>=4: rows.append([k,t]+[np.median(X[q,i]) for i in range(len(feats))])
R=np.array(rows); tr=R[:,1].astype(int); ICC={}; persist={}
for i,f in enumerate(feats):
    v=R[:,2+i].copy()
    for k in range(nf): s=R[:,0]==k; v[s]-=v[s].mean()           # remove frame effect
    u,cnt=np.unique(tr,return_counts=True); keep=np.isin(tr,u[cnt>=4]); vv,tt=v[keep],tr[keep]
    g=[vv[tt==a] for a in np.unique(tt)]; n0=np.mean([len(x) for x in g]); MSB=sum(len(x)*(x.mean()-vv.mean())**2 for x in g)/(len(g)-1); MSW=sum(((x-x.mean())**2).sum() for x in g)/(len(vv)-len(g))
    ICC[f]=float((MSB-MSW)/(MSB+(n0-1)*MSW))
    # lag correlation: same track, consecutive observations
    a=[];b=[]
    for t in np.unique(tt):
        x=R[keep][tt==t]; o=np.argsort(x[:,0]); x=x[o]; vx=vv[tt==t][o]
        for j in range(len(x)-1):
            if x[j+1,0]-x[j,0]<=2: a.append(vx[j]);b.append(vx[j+1])
    persist[f]=float(np.corrcoef(a,b)[0,1])
ntr=int(np.sum(np.unique(tr,return_counts=True)[1]>=4))
# spatial autocorrelation scale of each feature (pixel level): correlation length from radial autocorr, frame-averaged
def corrlen(f_i):
    Ls=[]
    for k in range(0,nf,4):
        s=np.where(fr==k)[0]; ny=int(d['iy'][s].max())+1; nx=int(d['ix'][s].max())+1; G=X[s,f_i].reshape(ny,nx).copy(); M=np.isfinite(G)&m[s].reshape(ny,nx)
        G[~M]=0; G[M]-=G[M].mean(); F1=np.fft.fft2(G,(2*ny,2*nx)); W=np.fft.fft2(M.astype(float),(2*ny,2*nx))
        ac=np.fft.ifft2(F1*np.conj(F1)).real/np.maximum(np.fft.ifft2(W*np.conj(W)).real,1); ac/=ac[0,0]
        yy,xx=np.meshgrid(np.fft.fftfreq(2*ny)*2*ny*PY,np.fft.fftfreq(2*nx)*2*nx*PX,indexing='ij'); r=np.hypot(yy,xx)
        rb=np.arange(0,15,0.5); prof=[ac[(r>=rb[j])&(r<rb[j]+0.5)].mean() for j in range(len(rb))]
        j=np.argmax(np.array(prof)<1/np.e); Ls.append(rb[j] if j>0 else np.nan)
    return float(np.nanmedian(Ls))
CL={f:corrlen(i) for i,f in enumerate(feats)}
out=dict(profiles=prof,comp=comp,ICC=ICC,persist_lag=persist,n_tracks_ge4=ntr,n_cellobs=len(R),corr_len_um=CL)
json.dump(out,open('../facts_cells.json','w'),indent=1); print(json.dumps({k:out[k] for k in ['ICC','persist_lag','n_tracks_ge4','n_cellobs','corr_len_um']},indent=1))
for f in feats: print(f,[round(p[0],2) for p in prof[f]])
fig,ax=plt.subplots(figsize=(4.4,2.6)); xs=np.arange(len(feats))
ax.bar(xs-.2,[ICC[f] for f in feats],.4,label='ICC (share of variance between cells)');ax.bar(xs+.2,[persist[f] for f in feats],.4,label='r, same cell, next frame')
ax.set_xticks(xs,[names[f] for f in feats],fontsize=8);ax.set_ylim(0,1);ax.legend(frameon=False,fontsize=7);plt.savefig('../fig/cell_identity.pdf');plt.close()
