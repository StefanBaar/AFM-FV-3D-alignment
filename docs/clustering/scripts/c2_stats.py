import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({'font.size':9,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
d=dict(np.load('../pix.npz')); F={}
ok=d['ok'].copy()
for f in ['slope','pull','nrup','teth','topo']: ok&=np.isfinite(d[f])
np.save('../ok.npy',ok)
feat=['logE','alpha','logd','Fadh','logW','Ladh','pull','nrup','teth','slope','topo']
lab={'logE':r'$\log_{10}E_{1\,\mathrm{ms}}$','alpha':r'$\alpha$','logd':r'$\log_{10}\delta$','Fadh':r'$F_\mathrm{adh}$','logW':r'$\log_{10}W_\mathrm{adh}$','Ladh':r'$L_\mathrm{adh}$','pull':r'pull-off $z$','nrup':'# ruptures','teth':'tether','slope':'slope','topo':'height'}
# within-frame Spearman, averaged over frames (removes global drift)
M=np.zeros((len(feat),len(feat))); nf=int(d['frame'].max())+1
for k in range(nf):
    m=ok&(d['frame']==k); X=np.column_stack([d[f][m] for f in feat]); M+=spearmanr(X).statistic/nf
F['spearman_within']=dict(feat=feat,M=np.round(M,3).tolist())
fig,ax=plt.subplots(figsize=(5.4,4.6)); im=ax.imshow(M,cmap='RdBu_r',vmin=-1,vmax=1)
ax.set_xticks(range(len(feat)),[lab[f] for f in feat],rotation=60,ha='right'); ax.set_yticks(range(len(feat)),[lab[f] for f in feat])
for i in range(len(feat)):
    for j in range(len(feat)): ax.text(j,i,f'{M[i,j]:.2f}'.replace('0.','.'),ha='center',va='center',fontsize=6.5,color='w' if abs(M[i,j])>.55 else 'k')
plt.colorbar(im,fraction=.046,label='Spearman ρ (within frame, mean of 41)'); plt.savefig('../fig/corr_matrix.pdf'); plt.close()
# delta vs E: Hertz at fixed force predicts slope -2/3
m=ok; X=np.column_stack([np.ones(m.sum()),d['logE'][m],d['alpha'][m]]); y=d['logd'][m]
# demean per frame
fr=d['frame'][m]
def dm(v): out=v.copy(); 
for _ in [0]: pass
def demean(v):
    o=v.astype(float).copy()
    for k in range(nf): s=fr==k; o[s]-=o[s].mean()
    return o
Xd=np.column_stack([demean(d['logE'][m]),demean(d['alpha'][m])]); yd=demean(y)
b1=np.linalg.lstsq(Xd[:,:1],yd,rcond=None)[0]; r2_1=1-np.var(yd-Xd[:,:1]@b1)/np.var(yd)
b2=np.linalg.lstsq(Xd,yd,rcond=None)[0]; res=yd-Xd@b2; r2_2=1-np.var(res)/np.var(yd)
sl=demean(d['slope'][m]); b3=np.linalg.lstsq(np.column_stack([Xd,sl]),yd,rcond=None)[0]; r2_3=1-np.var(yd-np.column_stack([Xd,sl])@b3)/np.var(yd)
F['delta_fit']=dict(slope_logE=float(b1[0]),R2_logE=float(r2_1),coef_logE_alpha=b2.tolist(),R2_logE_alpha=float(r2_2),R2_plus_slope=float(r2_3),coef_with_slope=b3.tolist())
dstar=np.full(len(ok),np.nan); dstar[m]=res; d['dstar']=dstar
fig,ax=plt.subplots(1,2,figsize=(8,3.2))
sel=np.random.default_rng(0).choice(m.sum(),20000,replace=False)
ax[0].hexbin(Xd[:,0],yd,gridsize=60,bins='log',cmap='Greys',mincnt=1); xx=np.linspace(-1,1,10)
ax[0].plot(xx,b1[0]*xx,'C3',label=f'fit: slope {b1[0]:.2f}, $R^2$={r2_1:.2f}'); ax[0].plot(xx,-2/3*xx,'C0--',label='Hertz, fixed force: −2/3')
ax[0].set_xlim(-.8,.8);ax[0].set_ylim(-.5,.5);ax[0].set_xlabel(r'$\log_{10}E_{1\,ms}$ (frame-demeaned)');ax[0].set_ylabel(r'$\log_{10}\delta$ (frame-demeaned)');ax[0].legend(frameon=False,fontsize=7.5)
ax[1].hexbin(d['slope'][m],res,gridsize=60,bins='log',cmap='Greys',mincnt=1);ax[1].set_xlabel('local slope (°)');ax[1].set_ylabel(r'$\delta^*$ = residual $\log_{10}\delta$')
rs=spearmanr(d['slope'][m],res).statistic; ax[1].set_title(f'Spearman ρ = {rs:.2f}',fontsize=9); F['dstar_slope_rho']=float(rs)
plt.tight_layout(); plt.savefig('../fig/delta_vs_E.pdf'); plt.close()
# adhesion block PCA (frame-demeaned, standardized)
A=['Fadh','logW','Ladh','pull','nrup','teth']
Z=np.column_stack([demean(d[a][m]) for a in A]); Z=(Z)/Z.std(0)
w,V=np.linalg.eigh(np.cov(Z.T)); o=np.argsort(w)[::-1]; w,V=w[o],V[:,o]
F['adh_pca']=dict(feat=A,var=(w/w.sum()).round(3).tolist(),pc1=V[:,0].round(2).tolist(),pc2=V[:,1].round(2).tolist())
# slope dependence of every feature
F['rho_slope']={f:float(np.mean([spearmanr(d[f][ok&(d['frame']==k)],d['slope'][ok&(d['frame']==k)]).statistic for k in range(nf)])) for f in ['logE','alpha','logd','Fadh','logW','Ladh','pull','teth']}
F['n_valid']=int(ok.sum()); F['n_pix']=int(len(ok)); F['tether_frac']=float(d['teth'][ok].mean()); F['nrup_mean']=float(d['nrup'][ok].mean())
np.save('../dstar.npy',dstar); json.dump(F,open('../facts_stats.json','w'),indent=1); print(json.dumps({k:v for k,v in F.items() if k!='spearman_within'},indent=1))
