import numpy as np, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import common as C
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'savefig.bbox':'tight','pdf.fonttype':42})
Z=C.Z; v=Z['valid']; px,py=Z['pixel_um']; ny,nx=Z['topo_um'].shape; ext=[0,nx*px,0,ny*py]
m=lambda a: np.where(v,a,np.nan)
pan=[('height (µm)',Z['topo_um'],'cividis',None),('slope (°)',Z['slope_deg'],'Greys',None),('E$_1$ at t$_0$ = 1 s (Pa)',m(Z['E1_Pa']),'viridis','log'),
     ('α',m(Z['alpha']),'magma',None),('E(2 ms) (Pa)',m(Z['E_at_2ms_Pa']),'viridis','log'),('E(10 ms) (Pa)',m(Z['E_at_10ms_Pa']),'viridis','log'),
     ('δ$_{max}$ (µm)',m(Z['delta_max_um']),'plasma',None),('R²',Z['R2'],'Greens',None),('E$_1$ normal / vertical',m(Z['E1_Pa']/Z['E1_vertical_Pa']),'coolwarm',None)]
fig,ax=plt.subplots(3,3,figsize=(12,10.5))
for a,(tt,d,cm,sc) in zip(ax.ravel(),pan):
    f=d[np.isfinite(d)]
    kw=dict(norm=LogNorm(*np.percentile(f,[2,98]))) if sc=='log' else dict(vmin=np.percentile(f,2),vmax=np.percentile(f,98))
    if tt.startswith('R'): kw=dict(vmin=0.9,vmax=1.0)
    h=a.imshow(d,origin='lower',extent=ext,cmap=cm,interpolation='nearest',**kw); a.set_title(tt); fig.colorbar(h,ax=a,shrink=.8); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig('fig/maps.pdf',dpi=150); plt.close()
lE1=np.log10(Z['E1_Pa'][v]); lE10=np.log10(Z['E_at_10ms_Pa'][v]); al=Z['alpha'][v]
fig,ax=plt.subplots(1,2,figsize=(9,3.4))
ax[0].scatter(al,lE1,s=2,alpha=.4,c='k'); ax[0].set_xlabel('α'); ax[0].set_ylabel('log$_{10}$ E$_1$ (Pa)'); ax[0].set_title(f'E$_1$ (1 s) vs α: r = {np.corrcoef(al,lE1)[0,1]:.2f}')
ax[1].scatter(al,lE10,s=2,alpha=.4,c='k'); ax[1].set_xlabel('α'); ax[1].set_ylabel('log$_{10}$ E(10 ms) (Pa)'); ax[1].set_title(f'E(10 ms) vs α: r = {np.corrcoef(al,lE10)[0,1]:.2f}')
plt.tight_layout(); plt.savefig('fig/coupling.pdf'); plt.close()
f=json.load(open('facts.json'))
f['maps']=dict(n=int(v.size),valid=int(v.sum()),E1_med=float(np.median(Z['E1_Pa'][v])),E1_iqr=[float(x) for x in np.percentile(Z['E1_Pa'][v],[25,75])],
  al_med=float(np.median(al)),al_iqr=[float(x) for x in np.percentile(al,[25,75])],E2_med=float(np.median(Z['E_at_2ms_Pa'][v])),E10_med=float(np.median(Z['E_at_10ms_Pa'][v])),
  d_med=float(np.median(Z['delta_max_um'][v])),R2_med=float(np.nanmedian(Z['R2'])),r_E1_al=float(np.corrcoef(al,lE1)[0,1]),r_E10_al=float(np.corrcoef(al,lE10)[0,1]),
  ratio_norm_vert_med=float(np.nanmedian((Z['E1_Pa']/Z['E1_vertical_Pa'])[v])),slope_med=float(np.nanmedian(Z['slope_deg'])),
  Fmax_med=float(np.nanmedian(Z['Fmax_nN'])))
json.dump(f,open('facts.json','w'),indent=1); print(f['maps'])
