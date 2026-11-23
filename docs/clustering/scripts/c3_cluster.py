import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score as ARI
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
d=dict(np.load('../pix.npz')); ok=np.load('../ok.npy'); d['dstar']=np.load('../dstar.npy'); nf=int(d['frame'].max())+1; fr=d['frame']
F=json.load(open('../facts_stats.json'))
def demean(v,m):
    o=np.full(len(v),np.nan); 
    for k in range(nf):
        s=m&(fr==k); x=v[s]; o[s]=(x-np.median(x))/(1.4826*np.median(np.abs(x-np.median(x)))+1e-12)
    return o
def resid_slope(v,m):           # remove a per-frame quadratic trend in slope
    o=np.full(len(v),np.nan)
    for k in range(nf):
        s=m&(fr==k); S=d['slope'][s]; A=np.column_stack([np.ones(s.sum()),S,S**2]); c=np.linalg.lstsq(A,v[s],rcond=None)[0]; o[s]=v[s]-A@c
    return o
m=ok&np.isfinite(d['dstar'])
A=['Fadh','logW','Ladh','pull','nrup','teth']; pc=np.array(F['adh_pca']['pc1']); pc2=np.array(F['adh_pca']['pc2'])
Za=np.column_stack([demean(d[a],m) for a in A]); Za=Za/np.nanstd(Za[m],0)
d['adh1']=-(Za@pc); d['adh2']=Za@pc2           # adh1: overall adhesion (sign: + = more adhesive); adh2: + = short-range strong, - = long-range / tethers
feats=['logE','alpha','dstar','adh1','adh2']; names={'logE':'log E','alpha':'α','dstar':'δ*','adh1':'adh. strength','adh2':'adh. short vs long'}
def build(resid):
    cols=[]
    for f in feats:
        v=d[f].astype(float); v=resid_slope(v,m) if resid else v; cols.append(demean(v,m))
    return np.column_stack(cols)
res={}
rng=np.random.default_rng(1); idx=np.where(m)[0]; sub=rng.choice(idx,30000,replace=False)
for tag,resid in [('raw',False),('slope_resid',True)]:
    X=build(resid); Xs=X[sub]; bic=[]
    for k in range(2,10):
        g=GaussianMixture(k,covariance_type='full',n_init=3,random_state=0).fit(Xs); bic.append(g.bic(Xs))
    res[tag]=dict(X=X,bic=bic)
# choose k: smallest k within 1% of BIC range from the minimum improvement -> report curve; use k=5 for interpretability if elbow
bic=np.array(res['slope_resid']['bic']); kk=np.arange(2,10)
rel=(bic-bic.min())/(bic.max()-bic.min()); K=5   # stability peak (c3b_stab.py): BIC keeps falling, so k is chosen by reproducibility
print('BIC rel',np.round(rel,3),'-> K',K)
out={'K':K,'bic_raw':res['raw']['bic'],'bic_resid':res['slope_resid']['bic']}
labs={}
for tag in res:
    X=res[tag]['X']; g=GaussianMixture(K,covariance_type='full',n_init=5,random_state=0).fit(X[sub]); L=np.full(len(fr),-1); L[m]=g.predict(X[m])
    # order clusters by mean log E
    mu=np.array([np.nanmean(X[L==j,0]) for j in range(K)]); o=np.argsort(mu); rmap=np.empty(K,int); rmap[o]=np.arange(K); L[m]=rmap[L[m]]
    labs[tag]=L; res[tag]['g']=g
out['ARI_raw_vs_resid']=float(ARI(labs['raw'][m],labs['slope_resid'][m]))
# stability: refit on other subsamples
L0=labs['slope_resid']; X=res['slope_resid']['X']; aris=[]
for s in range(5):
    sb=np.random.default_rng(10+s).choice(idx,30000,replace=False); g=GaussianMixture(K,covariance_type='full',n_init=3,random_state=s).fit(X[sb]); aris.append(ARI(L0[m],g.predict(X[m])))
out['ARI_resample']=[float(a) for a in aris]
# cluster profiles (in robust z units) + fraction + slope
prof=[];
for j in range(K):
    s=L0==j; prof.append(dict(frac=float(s[m].mean()),z={f:float(np.nanmedian(X[s,i])) for i,f in enumerate(feats)},
        E1ms_Pa=float(10**np.median(d['logE'][s])),alpha=float(np.median(d['alpha'][s])),delta_um=float(10**np.median(d['logd'][s])),
        Fadh=float(np.median(d['Fadh'][s])),Ladh=float(np.median(d['Ladh'][s])),teth=float(d['teth'][s].mean()),slope=float(np.median(d['slope'][s])),
        boundary_frac=float((d['cell'][s]==0).mean())))
out['profiles']=prof; out['boundary_frac_all']=float((d['cell'][m]==0).mean())
# spatial coherence: same-label 4-neighbour fraction vs chance
coh=[];chance=[]
for k in range(nf):
    s=fr==k; ny=int(d['iy'][s].max())+1; nx=int(d['ix'][s].max())+1; G=L0[s].reshape(ny,nx)
    a=np.concatenate([(G[:,1:]==G[:,:-1])[(G[:,1:]>=0)&(G[:,:-1]>=0)],(G[1:]==G[:-1])[(G[1:]>=0)&(G[:-1]>=0)]]); coh.append(a.mean())
    p=np.bincount(G[G>=0],minlength=K)/np.sum(G>=0); chance.append((p**2).sum())
out['coherence']=float(np.mean(coh)); out['coherence_chance']=float(np.mean(chance))
# temporal fractions
fracs=np.array([[np.mean(L0[(fr==k)&m]==j) for j in range(K)] for k in range(nf)]); tt=np.array([d['t_h'][fr==k][0] for k in range(nf)])
np.savez('../labels.npz',L=L0,Lraw=labs['raw'],X=X,m=m,feats=feats,fracs=fracs,tt=tt,adh1=d['adh1'],adh2=d['adh2'])
json.dump(out,open('../facts_cluster.json','w'),indent=1)
print(json.dumps({k:v for k,v in out.items() if k not in('profiles',)},indent=1)); [print(j,p) for j,p in enumerate(prof)]
# ---- figures
cols=plt.get_cmap('tab10').colors
fig,ax=plt.subplots(figsize=(3.6,2.6)); ax.plot(kk,(np.array(res['raw']['bic'])-min(res['raw']['bic']))/1e3,'o-',label='raw features');ax.plot(kk,(bic-bic.min())/1e3,'s-',label='slope-residualised');ax.axvline(K,c='gray',ls=':')
ax.set_xlabel('number of clusters k');ax.set_ylabel('ΔBIC (×10³)');ax.legend(frameon=False);plt.savefig('../fig/bic.pdf');plt.close()
fig,ax=plt.subplots(figsize=(5.6,2.8)); w=0.8/K
for j in range(K): ax.bar(np.arange(len(feats))+(j-(K-1)/2)*w,[prof[j]['z'][f] for f in feats],w,color=cols[j],label=f'C{j} ({100*prof[j]["frac"]:.0f}%)')
ax.axhline(0,c='k',lw=.6);ax.set_xticks(range(len(feats)),[names[f] for f in feats]);ax.set_ylabel('median (robust z, per frame)');ax.legend(frameon=False,fontsize=7,ncol=3);plt.savefig('../fig/profiles.pdf');plt.close()
fig,ax=plt.subplots(figsize=(4.2,2.6)); ax.stackplot(tt,fracs.T,colors=cols[:K],labels=[f'C{j}' for j in range(K)]);ax.set_xlim(tt[0],tt[-1]);ax.set_ylim(0,1);ax.set_xlabel('time (h)');ax.set_ylabel('fraction of pixels');ax.legend(frameon=False,fontsize=7,ncol=K,loc='upper center',bbox_to_anchor=(.5,1.18));plt.savefig('../fig/fractions_time.pdf');plt.close()
# pair plot of E-alpha and adhesion coloured by cluster
fig,ax=plt.subplots(1,3,figsize=(9,2.9)); sb=np.random.default_rng(3).choice(idx,15000,replace=False)
for a,(i,j) in zip(ax,[(0,1),(0,3),(3,4)]):
    a.scatter(X[sb,i],X[sb,j],s=1.5,c=[cols[c] for c in L0[sb]],rasterized=True,lw=0);a.set_xlabel(names[feats[i]]+' (z)');a.set_ylabel(names[feats[j]]+' (z)');a.set_xlim(-4,4);a.set_ylim(-4,4)
plt.tight_layout();plt.savefig('../fig/scatter_clusters.pdf',dpi=220);plt.close()
