import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage as ndi
from scipy.stats import spearmanr
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'savefig.bbox':'tight','pdf.fonttype':42})
z=np.load('/home/claude/fullv/home_claude_data.npz'); v=z['valid']; px,py=z['pixel_um']; ny,nx=v.shape; ext=[0,nx*px,0,ny*py]
cells=np.load('/home/claude/seg2/maps/cells/data_20260515_0.25nN_sample_ciona_data_213626.npy')
V=[('main','CP fitted','',  '#d62839'),('vert','CP fitted, vertical','_vertical','#e07b39'),('lab','lab CP (toe start)','_labcp','#2a9d55'),('sq','CP fitted + squeeze','_sq','#2a6fbb')]
def E(suf,t): return z[f'E_at_{t}ms{suf}_Pa'] if t!='1s' else z[f'E1{suf}_Pa']
def A(suf): return z['alpha'+suf] if suf else z['alpha']
# region masks: junction = label 0 (watershed lines); interior = label>0 and >=1 px from any other label
jun=v&(cells==0)
inter=np.zeros_like(v)
for l in np.unique(cells[cells>0]):
    m=cells==l; inter|=ndi.binary_erosion(m)&v
facts={'n_valid':int(v.sum()),'n_junction':int(jun.sum()),'n_interior':int(inter.sum())}
rows=[]
for k,lab,suf,c in V:
    r={'key':k,'label':lab,'alpha_med':float(np.nanmedian(A(suf)[v]))}
    for t in ('1','10','1s'):
        e=E(suf,t); le=np.log10(np.where(v&(e>0),e,np.nan))
        r[f'E{t}_med']=float(np.nanmedian(10**le)); r[f'E{t}_cv']=float(np.subtract(*np.nanpercentile(le,[75,25]))/1.349*np.log(10))   # robust SD of ln E
        r[f'E{t}_junc_over_int']=float(10**(np.nanmedian(le[jun])-np.nanmedian(le[inter])))
        r[f'r_alpha_logE{t}']=float(np.corrcoef(A(suf)[np.isfinite(le)],le[np.isfinite(le)])[0,1])
    a=A(suf); r['alpha_sd']=float(np.subtract(*np.nanpercentile(a[v],[75,25]))/1.349); r['alpha_junc_minus_int']=float(np.nanmedian(a[jun])-np.nanmedian(a[inter]))
    rows.append(r)
e=z['E_elastic_Pa']; le=np.log10(np.where(v&(e>0),e,np.nan))
rows.append({'key':'el','label':'elastic (α = 0)','E1_med':float(np.nanmedian(10**le)),'E1_cv':float(np.subtract(*np.nanpercentile(le,[75,25]))/1.349*np.log(10)),'E1_junc_over_int':float(10**(np.nanmedian(le[jun])-np.nanmedian(le[inter])))})
facts['rows']=rows
# ---- maps (common scales)
fig,ax=plt.subplots(3,5,figsize=(19,10.5))
for j,(k,lab,suf,c) in enumerate(V):
    ax[0,j].imshow(np.where(v,A(suf),np.nan),origin='lower',extent=ext,cmap='magma',vmin=.3,vmax=.65,interpolation='nearest'); ax[0,j].set_title('α · '+lab)
    h1=ax[1,j].imshow(np.where(v,E(suf,'1'),np.nan),origin='lower',extent=ext,cmap='viridis',norm=LogNorm(150,4000),interpolation='nearest'); ax[1,j].set_title('E(1 ms) · '+lab)
    h2=ax[2,j].imshow(np.where(v,E(suf,'1s'),np.nan),origin='lower',extent=ext,cmap='viridis',norm=LogNorm(3,300),interpolation='nearest'); ax[2,j].set_title('E₁ (1 s) · '+lab)
ha=ax[0,0].images[0]
ax[0,4].imshow(np.where(v,np.where(jun,1.0,np.where(inter,0.0,0.5)),np.nan),origin='lower',extent=ext,cmap='gray_r',interpolation='nearest'); ax[0,4].set_title('segmentation: junction (black), interior (white)')
ax[1,4].imshow(np.where(v,z['E_elastic_Pa'],np.nan),origin='lower',extent=ext,cmap='viridis',norm=LogNorm(150,4000),interpolation='nearest'); ax[1,4].set_title('E · elastic (α = 0)')
ax[2,4].axis('off')
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.colorbar(ha,ax=ax[0,:4],shrink=.85,label='α'); fig.colorbar(h1,ax=ax[1,:],shrink=.85,label='E(1 ms) (Pa)'); fig.colorbar(h2,ax=ax[2,:4],shrink=.85,label='E₁ at 1 s (Pa)')
plt.savefig('fig/maps_common.png',dpi=110); plt.close()
# ---- normalised maps: each map as robust z-score -> pattern/contrast comparison
fig,ax=plt.subplots(2,4,figsize=(15,7))
for j,(k,lab,suf,c) in enumerate(V):
    for i,(X,tt) in enumerate([(A(suf),'α'),(np.log10(np.where(E(suf,'1s')>0,E(suf,'1s'),np.nan)),'log E₁ (1 s)')]):
        X=np.where(v,X,np.nan); zz=(X-np.nanmedian(X))/(np.subtract(*np.nanpercentile(X,[75,25]))/1.349)
        h=ax[i,j].imshow(zz,origin='lower',extent=ext,cmap='coolwarm',vmin=-2.5,vmax=2.5,interpolation='nearest'); ax[i,j].set_title(f'{tt} · {lab}\n(own median / robust SD)',fontsize=8.5); ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
fig.colorbar(h,ax=ax,shrink=.8,label='robust z-score'); plt.savefig('fig/maps_normalised.png',dpi=110); plt.close()
# ---- rank agreement between variants (Spearman) for alpha, E(1 ms), E1
def rmat(get):
    M=np.zeros((4,4))
    for a_,(k1,_,s1,_) in enumerate(V):
        for b_,(k2,_,s2,_) in enumerate(V):
            x,y=get(s1),get(s2); m=v&np.isfinite(x)&np.isfinite(y); M[a_,b_]=spearmanr(x[m],y[m])[0]
    return M
mats=[('α',A),('E(1 ms)',lambda s:E(s,'1')),('E₁ (1 s)',lambda s:E(s,'1s'))]
fig,ax=plt.subplots(1,3,figsize=(15,4.3)); facts['spearman']={}
for a,(tt,get) in zip(ax,mats):
    M=rmat(get); facts['spearman'][tt]=M.round(3).tolist()
    h=a.imshow(M,vmin=0,vmax=1,cmap='Blues')
    for i in range(4):
        for j in range(4): a.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',color='white' if M[i,j]>.6 else 'k')
    a.set_xticks(range(4)); a.set_xticklabels([x[1] for x in V],rotation=30,ha='right',fontsize=8); a.set_yticks(range(4)); a.set_yticklabels([x[1] for x in V],fontsize=8); a.set_title(f'rank correlation of the {tt} maps')
plt.tight_layout(); plt.savefig('fig/spearman.pdf'); plt.close()
# ---- contrast bars
fig,ax=plt.subplots(1,3,figsize=(15,3.6)); labs=[r_['label'] for r_ in rows[:4]]; cols=[x[3] for x in V]
xx=np.arange(4); w=0.27
for i,t in enumerate(('1','10','1s')):
    ax[0].bar(xx+(i-1)*w,[r_[f'E{t}_cv'] for r_ in rows[:4]],w,label={'1':'E(1 ms)','10':'E(10 ms)','1s':'E₁ (1 s)'}[t])
    ax[1].bar(xx+(i-1)*w,[r_[f'E{t}_junc_over_int'] for r_ in rows[:4]],w,label={'1':'E(1 ms)','10':'E(10 ms)','1s':'E₁ (1 s)'}[t])
ax[0].axhline(rows[4]['E1_cv'],c='k',ls=':',label='elastic'); ax[1].axhline(rows[4]['E1_junc_over_int'],c='k',ls=':',label='elastic'); ax[1].axhline(1,c='grey',lw=.5)
ax[0].set_title('spatial contrast: robust SD of ln E across the map'); ax[1].set_title('junction / interior (median E ratio)'); ax[1].set_ylim(0.95,1.22)
ax[2].bar(xx-w/2,[r_['alpha_sd'] for r_ in rows[:4]],w,label='robust SD of α',color=cols); ax[2].bar(xx+w/2,[r_['alpha_junc_minus_int'] for r_ in rows[:4]],w,label='α(junction) − α(interior)',color=cols,alpha=.5,hatch='//')
ax[2].set_title('α contrast'); ax[2].legend(frameon=False,fontsize=7.5)
for a in ax: a.set_xticks(xx); a.set_xticklabels(labs,rotation=20,ha='right',fontsize=8)
ax[0].legend(frameon=False,fontsize=7,loc='upper left'); ax[1].legend(frameon=False,fontsize=7,loc='upper left',ncol=2)
plt.tight_layout(); plt.savefig('fig/contrast.pdf'); plt.close()
# ---- E over alpha: rows variants, cols reference time
fig,ax=plt.subplots(4,3,figsize=(12,12),sharex=True)
for i,(k,lab,suf,c) in enumerate(V):
    a_=A(suf)
    for j,t in enumerate(('1','10','1s')):
        e=E(suf,t); m=v&(e>0)&np.isfinite(a_)
        ax[i,j].scatter(a_[m&~jun&~inter],e[m&~jun&~inter],s=2,c='#bbb',alpha=.5,label='other')
        ax[i,j].scatter(a_[m&inter],e[m&inter],s=2,c=c,alpha=.45,label='cell interior')
        ax[i,j].scatter(a_[m&jun],e[m&jun],s=3,c='k',alpha=.6,label='junction')
        r=np.corrcoef(a_[m],np.log10(e[m]))[0,1]
        ax[i,j].set_yscale('log'); ax[i,j].set_title(f'{lab}: {"E(1 ms)" if t=="1" else "E(10 ms)" if t=="10" else "E₁ (1 s)"}   r = {r:+.2f}',fontsize=8.5)
        if j==0: ax[i,j].set_ylabel('E (Pa)')
        if i==3: ax[i,j].set_xlabel('α')
ax[0,0].legend(frameon=False,fontsize=7,markerscale=3); ax[0,0].set_xlim(0.25,0.7)
plt.tight_layout(); plt.savefig('fig/E_vs_alpha.png',dpi=120); plt.close()
# ---- correlation vs reference time, all variants: embryo region (solid) and whole map (dashed)
reg=v&(z['E_at_1ms_Pa']>500)
T=np.logspace(-4,0.3,120); fig,ax=plt.subplots(figsize=(6.8,3.8)); facts['tstar_ms']={}; facts['tstar_ms_embryo']={}
for k,lab,suf,c in V:
    a_=A(suf); lE1=np.log10(np.where(z[f'E1{suf}_Pa']>0,z[f'E1{suf}_Pa'],np.nan))
    for msk,ls,key in ((v,'--','tstar_ms'),(reg,'-','tstar_ms_embryo')):
        m=msk&np.isfinite(lE1)&np.isfinite(a_)
        cc=np.array([np.corrcoef(a_[m],lE1[m]-a_[m]*np.log10(tt))[0,1] for tt in T]); ax.semilogx(T,cc,c=c,ls=ls,label=lab if ls=='-' else None)
        facts[key][k]=float(T[np.argmin(np.abs(cc))]*1e3) if (cc.min()<0<cc.max()) else None
ax.plot([],[],c='k',ls='-',label='embryo surface (E(1 ms) > 500 Pa)'); ax.plot([],[],c='k',ls='--',label='whole map')
ax.axhline(0,c='k',lw=.5); [ax.axvline(x,c='grey',ls=':',lw=.8) for x in (1e-3,1e-2,1)]
ax.axvspan(2e-4,2e-2,color='grey',alpha=.08); ax.text(2.3e-4,0.85,'measured time range',fontsize=7.5)
ax.set_xlabel('reference time t (s)'); ax.set_ylabel('corr( log E(t), α )'); ax.set_ylim(-1,1); ax.legend(frameon=False,fontsize=7,ncol=2,loc='lower left')
plt.tight_layout(); plt.savefig('fig/corr_vs_t.pdf'); plt.close()
# ---- E(t) fan (main vs squeeze vs lab): median and junction/interior medians
fig,ax=plt.subplots(1,3,figsize=(14,3.6),sharey=True); tt=np.logspace(-4,0.3,100)
for a,(k,lab,suf,c) in zip(ax,[V[0],V[2],V[3]]):
    a_=A(suf); E1=z[f'E1{suf}_Pa']
    for msk,cc,ll in ((inter,c,'interior'),(jun,'k','junction')):
        aa=np.nanmedian(a_[msk]); e1=np.nanmedian(np.log10(E1[msk&(E1>0)]))
        idx=np.where(msk&(E1>0))[0] if False else None
        sel=np.argwhere(msk&(E1>0)); 
        for q in sel[::15]: a.loglog(tt,E1[tuple(q)]*tt**(-a_[tuple(q)]),c=cc,alpha=.12,lw=.6)
        a.loglog(tt,10**e1*tt**(-aa),c=cc,lw=2.2,label=f'{ll}: median α {aa:.2f}')
    a.axvspan(2e-4,2e-2,color='grey',alpha=.08); a.set_title(lab); a.set_xlabel('t (s)'); a.legend(frameon=False,fontsize=7.5)
ax[0].set_ylabel('E(t) (Pa)'); plt.tight_layout(); plt.savefig('fig/fans.png',dpi=120); plt.close()
json.dump(facts,open('facts.json','w'),indent=1); print(json.dumps(facts,indent=1)[:4000])
# within-region correlations: embryo surface (main E(1 ms) > 500 Pa) vs soft region
reg=v&(z['E_at_1ms_Pa']>500); soft=v&(z['E_at_1ms_Pa']<=500)
facts['regions']={'n_embryo':int(reg.sum()),'n_soft':int(soft.sum()),'r':{}}
for k,lab,suf,c in V:
    a_=A(suf); facts['regions']['r'][k]={}
    for t in ('1','10','1s'):
        e=E(suf,t)
        for nm,msk in (('embryo',reg),('soft',soft)):
            m=msk&(e>0)&np.isfinite(a_); facts['regions']['r'][k][f'{nm}_{t}']=round(float(np.corrcoef(a_[m],np.log10(e[m]))[0,1]),3)
    facts['regions']['r'][k]['alpha_embryo']=round(float(np.nanmedian(a_[reg])),3); facts['regions']['r'][k]['alpha_soft']=round(float(np.nanmedian(a_[soft])),3)
    facts['regions']['r'][k]['E1ms_embryo']=round(float(np.nanmedian(E(suf,'1')[reg])),1); facts['regions']['r'][k]['E1ms_soft']=round(float(np.nanmedian(E(suf,'1')[soft])),1)
json.dump(facts,open('facts.json','w'),indent=1); print(json.dumps(facts['regions'],indent=0))
