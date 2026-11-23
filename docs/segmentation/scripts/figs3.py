import numpy as np, json, csv, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
import trace as T, segment_cells as S
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'savefig.bbox':'tight','pdf.fonttype':42})
DATA=Path('/home/claude/seg2/maps'); RUN=Path('/home/claude/seg2/run3')
fr=T.frames(); sc=json.loads((DATA/'cells_hr'/'scale.json').read_text())
tr={int(t['track']):t for t in csv.DictReader(open(RUN/'tracks.csv'))}
rows=list(csv.DictReader(open(RUN/'cells_table.csv')))
pos={(int(r['frame']),int(r['track'])):r for r in rows}
cand=[]
for t,d in tr.items():
    if d['division']=='confirmed':
        p=int(d['parent']); k0=int(d['first_frame'])
        nd=int(d['longest_reliable']); npar=int(tr[p]['longest_reliable'])
        if (k0-1,p) in pos and (k0,p) in pos:
            a0=float(pos[(k0-1,p)]['area_proj_um2']); a1=float(pos[(k0,p)]['area_proj_um2']); a2=float(pos[(k0,t)]['area_proj_um2'])
            cand.append((min(nd,npar),t,p,k0,a0,a1,a2))
cand.sort(reverse=True); print(cand[:8])
_,t,p,k0,a0,a1,a2=[c for c in cand if c[1]==479][0]
frs=[k for k in range(k0-2,k0+4) if 0<=k<len(fr)]
fig,ax=plt.subplots(1,len(frs),figsize=(2.8*len(frs),3.3))
for a,k in zip(ax,frs):
    f=fr[k]; L=np.load(DATA/'cells_hr'/(f['name']+'.npy')); sx,sy=sc[f['name']]
    A=f['v']['adh_force_nN'].astype(float); X=ndi.zoom(S.fill_nan(A),(L.shape[0]/A.shape[0],L.shape[1]/A.shape[1]),order=3); X=X-ndi.gaussian_filter(X,16); X[L<0]=np.nan
    ext=[0,L.shape[1]*sx,0,L.shape[0]*sy]
    a.imshow(X,origin='lower',extent=ext,cmap='gray',vmin=np.nanpercentile(X,2),vmax=np.nanpercentile(X,98))
    rg=np.zeros(L.shape+(4,)); rg[L==p]=[0.9,0.15,0.2,.5]; rg[L==t]=[0.1,0.45,0.95,.5]
    e=(L>=0)&((np.roll(L,1,0)!=L)|(np.roll(L,1,1)!=L)); rg[e|(L==0)]=[1,.85,0,1]
    a.imshow(rg,origin='lower',extent=ext,interpolation='nearest')
    c=pos.get((k,p)) or pos.get((k0,p)); cx,cy=float(c['x_raw_um']),float(c['y_raw_um'])
    a.set_xlim(cx-7,cx+7); a.set_ylim(cy-6,cy+6); a.set_xticks([]); a.set_yticks([])
    lab=[]; 
    if (k,p) in pos: lab.append(f'mother {p}: {float(pos[(k,p)]["area_proj_um2"]):.1f} µm²')
    if (k,t) in pos: lab.append(f'daughter {t}: {float(pos[(k,t)]["area_proj_um2"]):.1f} µm²')
    a.set_title(f'{f["time"]}\n'+'\n'.join(lab),fontsize=8)
plt.tight_layout(); plt.savefig('fig/division.pdf',dpi=200); plt.close()
json.dump(dict(t=t,p=p,k0=k0,time=fr[k0]['time'],a_parent_before=a0,a_parent_after=a1,a_daughter=a2,frames=[fr[k]['time'] for k in frs]),open('facts3.json','w'),indent=1)
