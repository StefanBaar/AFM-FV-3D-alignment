import numpy as np, json, csv, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from pathlib import Path
import trace as T, segment_cells as S, build_timelapse_viewer as BV
plt.rcParams.update({'font.size':9,'axes.titlesize':9.5,'savefig.bbox':'tight','pdf.fonttype':42})
DATA=Path('/home/claude/seg2/maps'); RUN=Path('/home/claude/seg2/run3')
fr=T.frames(); nw=2
pairs=BV.get_registration(fr,DATA,nw,verbose=False); _,G3=BV.chain(pairs)
sc=json.loads((DATA/'cells_hr'/'scale.json').read_text())
def native(f):
    ny,nx=f['v']['topo'].shape; yy,xx=np.mgrid[0:ny,0:nx]
    return np.stack([xx*f['px'],yy*f['py'],np.nan_to_num(f['v']['topo'])],-1)
def mapped(a,k):
    Grel=np.linalg.inv(G3[a])@G3[k]; N=native(fr[k]); P=N.reshape(-1,3)@Grel[:3,:3].T+Grel[:3,3]
    return native(fr[a])[...,:2], P[:,:2].reshape(N.shape[:2]+(2,)), Grel
cells=lambda f: np.load(DATA/'cells'/(f['name']+'.npy'))
hr=lambda f: np.load(DATA/'cells_hr'/(f['name']+'.npy'))
bnd=lambda f: np.load(DATA/'cell_boundary'/(f['name']+'.npy'))
facts={}
a,k=15,16; fa,fk=fr[a],fr[k]
xa,xk,Grel=mapped(a,k)
rf=S.refine_pair((xa.reshape(-1,2),bnd(fa).ravel(),xk.reshape(-1,2),bnd(fk).ravel(),0.5,3.0,4.0))
xk2=S.apply_refine(xk.reshape(-1,2),rf).reshape(xk.shape)
R=np.array(Grel[:3,:3]); ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
facts['pair']=dict(a=fa['time'],k=fk['time'],rel_angle_deg=float(ang),rel_t=[float(x) for x in Grel[:3,3]],
                   refine_deg=float(np.degrees(rf[0])),refine_dx=rf[1],refine_dy=rf[2],corr_before=rf[3],corr_after=rf[4])
def hr_edges_xy(f,G=None,refine=None):
    L=hr(f); sx,sy=sc[f['name']]; e=((L>0)&((np.roll(L,1,0)!=L)|(np.roll(L,1,1)!=L)))|(L==0)
    r,c=np.nonzero(e); x=c*sx; y=r*sy
    if G is None: return x,y
    T_=f['v']['topo']; z=ndi.map_coordinates(S.fill_nan(T_),[y/f['py'],x/f['px']],order=1)
    P=np.c_[x,y,z]@G[:3,:3].T+G[:3,3]; q=P[:,:2]
    if refine is not None: q=S.apply_refine(q,refine)
    return q[:,0],q[:,1]
fig,ax=plt.subplots(1,2,figsize=(11,4.9))
def rast(x,y,h=0.25,X=(0,45),Y=(0,37.5)):
    Hh,_,_=np.histogram2d(y,x,bins=[np.arange(Y[0],Y[1]+h,h),np.arange(X[0],X[1]+h,h)])
    return Hh>0
for j,(ttl,rfx) in enumerate([('registration only (rigid 3D)',None),('+ fine alignment on the boundary maps',rf)]):
    x0,y0=hr_edges_xy(fa); x1,y1=hr_edges_xy(fk,Grel,rfx)
    A0=rast(x0,y0); A1=rast(x1,y1)
    img=np.ones(A0.shape+(3,)); img[A0]=[0.15,0.35,0.85]; img[A1]=[0.9,0.2,0.25]; img[A0&A1]=[0.1,0.1,0.1]
    ax[j].imshow(img,origin='lower',extent=[0,45,0,37.5],interpolation='nearest')
    ax[j].set_xlim(10,32); ax[j].set_ylim(10,27); ax[j].set_xlabel('x (µm, scan plane of %s)'%fa['time']); ax[j].set_ylabel('y (µm)')
    ov=(A0&A1).sum()/max(A0.sum(),1)
    ax[j].set_title(f'{ttl}\nblue {fa["time"]}, red {fk["time"]}, black = both')
plt.tight_layout(); plt.savefig('fig/align.png',dpi=170); plt.close()
# IoU matrix and assignment for this pair
La=cells(fa); Lk=cells(fk)
pairs_,inter,ca,cb,st=S._match(La,xa,Lk,xk2,0.2,(fa['px'],fa['py']))
A=sorted(ca); Bl=sorted(cb); M=np.zeros((len(A),len(Bl)))
for (p,q),n in inter.items(): M[A.index(p),Bl.index(q)]=n/(ca[p]+cb[q]-n)
same=sum(1 for p,q in pairs_.items() if p==q)
facts['iou']=dict(n_a=len(A),n_k=len(Bl),n_linked=len(pairs_),median_iou=float(np.median(st)),
                  linked_same_track=same)
# choose 8 linked cells in a block for display
sel=[p for p in list(pairs_)[:200]][:8]; colsel=[pairs_[p] for p in sel]
extra=[q for q in Bl if q not in colsel and any(M[A.index(p),Bl.index(q)]>0 for p in sel)][:4]
cols=colsel+extra
sub=np.array([[M[A.index(p),Bl.index(q)] for q in cols] for p in sel])
fig,ax=plt.subplots(figsize=(6.2,4.4))
im=ax.imshow(sub,cmap='Blues',vmin=0,vmax=1)
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        if sub[i,j]>0: ax.text(j,i,f'{sub[i,j]:.2f}',ha='center',va='center',fontsize=7.5,color='white' if sub[i,j]>.5 else 'black')
    j=cols.index(pairs_[sel[i]]); ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec='#e63946',lw=2))
ax.set_xticks(range(len(cols))); ax.set_xticklabels([str(c) for c in cols],rotation=90,fontsize=7.5); ax.set_yticks(range(len(sel))); ax.set_yticklabels([str(c) for c in sel],fontsize=7.5)
ax.set_xlabel(f'cell (track ID) in {fk["time"]}'); ax.set_ylabel(f'cell (track ID) in {fa["time"]}')
ax.set_title('IoU of 8 cells with their overlapping cells; red = Hungarian assignment',fontsize=8.5)
plt.colorbar(im,fraction=.046,pad=.03,label='IoU'); plt.tight_layout(); plt.savefig('fig/iou.pdf'); plt.close()
# track statistics from the results
rows=list(csv.DictReader(open(RUN/'cells_table.csv'))); tr=list(csv.DictReader(open(RUN/'tracks.csv')))
nobs=np.array([int(t['n_observed']) for t in tr]); rel=np.array([int(t['longest_reliable']) for t in tr])
per=np.bincount([int(r['frame']) for r in rows],minlength=len(fr))
linked=np.zeros(len(fr))
byt={}
for r in rows: byt.setdefault(int(r['track']),[]).append(int(r['frame']))
for t,fs in byt.items():
    for x in fs:
        if x-1 in fs: linked[x]+=1
fig,ax=plt.subplots(1,3,figsize=(14,3.4))
ax[0].plot(per,'o-',ms=3,label='cells'); ax[0].plot(np.r_[np.nan,linked[1:]],'s-',ms=3,label='linked to the previous frame'); ax[0].set_xlabel('frame'); ax[0].set_ylabel('cells'); ax[0].legend(frameon=False)
bins=np.arange(1,max(nobs)+2)-.5
ax[1].hist(nobs,bins,color='#7f9cc9',label='observed frames per track'); ax[1].hist(rel,bins,color='#e63946',alpha=.6,label='longest reliable stretch'); ax[1].set_yscale('log'); ax[1].set_xlabel('frames'); ax[1].set_ylabel('tracks'); ax[1].legend(frameon=False)
st_=np.array([float(r['step_um']) for r in rows if r['step_um']!='']); ar_=np.array([float(r['area_change']) for r in rows if r['area_change']!=''])
ax[2].scatter(st_,ar_,s=3,alpha=.35,c='#444'); ax[2].axvline(3.0,c='#e63946',ls='--'); ax[2].axhline(1.8,c='#e63946',ls='--')
ax[2].set_xlabel('centroid step between observations (µm, 3D)'); ax[2].set_ylabel('area ratio (larger/smaller)'); ax[2].set_ylim(1,4); ax[2].set_xlim(0,10)
ax[2].set_title(f'{np.mean((st_<=3)&(ar_<=1.8))*100:.0f} % of links pass both limits')
plt.tight_layout(); plt.savefig('fig/trackstats.pdf'); plt.close()
facts['stats']=dict(tracks=len(tr),cells_total=len(rows),cells_per_frame_median=float(np.median(per)),linked_frac=float(linked[1:].sum()/per[1:].sum()),
   obs_ge=[int((nobs>=n).sum()) for n in (5,10,20,30)],rel_ge=[int((rel>=n).sum()) for n in (5,10,15,20)],longest=int(nobs.max()),longest_rel=int(rel.max()),
   link_ok=float(np.mean((st_<=3)&(ar_<=1.8))),med_step=float(np.median(st_)),med_area_ratio=float(np.median(ar_)),
   area_med=float(np.median([float(r['area_proj_um2']) for r in rows])),area_surf_med=float(np.median([float(r['area_surf_um2']) for r in rows])),
   eqd_med=float(np.median([float(r['eq_diam_um']) for r in rows])),
   div_conf=sum(t['division']=='confirmed' for t in tr),div_cand=sum(t['division']=='candidate' for t in tr),
   gap_links=None)
# lifelines as an image: rows = tracks (>= 10 observations), sorted by number of observations
tl=sorted([v for v in byt.values() if len(v)>=10],key=lambda v:(-len(v),min(v)))
Mx=np.ones((len(tl),len(fr),3))
for i,v in enumerate(tl):
    Mx[i,min(v):max(v)+1]=[0.85,0.85,0.85]; Mx[i,v]=[0.12,0.31,0.61]
fig,ax=plt.subplots(figsize=(10,4.2))
ax.imshow(Mx,aspect='auto',interpolation='nearest',extent=[-.5,len(fr)-.5,len(tl)-.5,-.5])
ax.set_xlabel('frame'); ax.set_ylabel('track (sorted by observations)')
ax.set_title(f'{len(tl)} tracks observed in ≥ 10 frames: blue = cell found, grey = bridged gap (≤ 3 frames missing)')
plt.tight_layout(); plt.savefig('fig/lifelines.pdf'); plt.close()
json.dump(facts,open('facts2.json','w'),indent=1,default=float); print(json.dumps(facts,indent=1,default=float))
