import numpy as np, json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
import trace as T, segment_cells as S
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'savefig.bbox':'tight'})
DATA=Path('/home/claude/seg2/maps'); OLD=Path('/home/claude/m3/maps/maps/cells')
fr=T.frames(); sc=json.loads((DATA/'cells_hr'/'scale.json').read_text())
sel=[f for f in fr if f['time'] in ('19:00:35','21:11:18')]
fig,ax=plt.subplots(2,2,figsize=(10.5,8.6)); res={}
for i,f in enumerate(sel):
    Lh=np.load(DATA/'cells_hr'/(f['name']+'.npy')); sx,sy=sc[f['name']]; Lo=np.load(OLD/(f['name']+'.npy'))
    A=f['v']['adh_force_nN'].astype(float); X=ndi.zoom(S.fill_nan(A),(Lh.shape[0]/A.shape[0],Lh.shape[1]/A.shape[1]),order=3); X=X-ndi.gaussian_filter(X,16); X[Lh<0]=np.nan
    ext=[0,Lh.shape[1]*sx,0,Lh.shape[0]*sy]
    # old labels upsampled (nearest) to the fine grid for drawing
    Lo_h=ndi.zoom(Lo,(Lh.shape[0]/Lo.shape[0],Lh.shape[1]/Lo.shape[1]),order=0)
    for j,(L,ttl) in enumerate([(Lo_h,'before (6 µm scales, all channels, 0.5 µm grid)'),(Lh,'now (D = 4 µm, adhesion channels, 0.25 µm grid)')]):
        a=ax[i,j]; a.imshow(X,origin='lower',extent=ext,cmap='gray',vmin=np.nanpercentile(X,2),vmax=np.nanpercentile(X,98))
        rg=np.zeros(L.shape+(4,)); e=(L>0)&((np.roll(L,1,0)!=L)|(np.roll(L,1,1)!=L)); rg[e]=[1,.15,.1,1]; rg[(L==0)]=[1,.15,.1,.45]
        a.imshow(rg,origin='lower',extent=ext,interpolation='nearest')
        n=len(np.unique(L[L>0])); ar=np.bincount(np.where(L>0,L,0).ravel())[1:]; ar=ar[ar>0]*sx*sy
        a.set_title(f'{f["time"]} {ttl}\n{n} cells, median area {np.median(ar):.1f} µm²',fontsize=9); a.set_xticks([]); a.set_yticks([])
        res[f'{f["time"]}_{j}']=(n,float(np.median(ar)))
plt.tight_layout(); plt.savefig('fig/oldnew.png',dpi=150); plt.close(); print(res); json.dump(res,open('facts4.json','w'))
